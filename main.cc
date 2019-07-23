
#include <stdio.h>
#include "xparameters.h"
#include "xaxidma.h"
#include "xfp_im2gemm.h"
#include "AxiTimerHelper.h"
//#include "fixedp.h"
#undef str
#include "ap_fixed.h"
#include "input.h"
#include "weight.h"


typedef ap_fixed<16,9> fp;
#define SIZE_ARR 100352+576


XFp_im2gemm im2gemm;
XFp_im2gemm_Config *im2gemm_cfg;
XAxiDma axiDMA;
XAxiDma_Config *axiDMA_cfg;

//DMA addresses
#define MEM_BASE_ADDR 0x00100000
#define TX_BUFFER_BASE (MEM_BASE_ADDR + 0x00100000)
#define RX_BUFFER_BASE (MEM_BASE_ADDR + 0x00300000)


float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col_cpu(float* data_im, int channels,  int height,  int width, int ksize,  int stride, int pad, float* data_col){

    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;

//#pragma omp parallel for
   for(i = 0; i < M; ++i){
       for(k = 0; k < K; ++k){
           register float A_PART = ALPHA*A[i*lda+k];
           for(j = 0; j < N; ++j){
               C[i*ldc+j] += A_PART*B[k*ldb+j];
           }
       }
   }
}

void scale_bias(float *output, float scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales;
            }
        }
    }
}

void add_bias(float *output, float biases, int batch, int n, int size)
{

    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases;
            }
        }
    }
}

unsigned int float_to_u32(float val){
    unsigned int result;
    union float_bytes{
        float v;
        unsigned char bytes[4];

    }data;

    data.v = val;

    result = (data.bytes[3]<<24)+(data.bytes[2]<<16)+(data.bytes[1]<<8)+(data.bytes[0]);

    return result;
}
/*
unsigned int u32_to_float(unsigned int val){

    union {
        float val_float;
        unsigned char bytes[4];

    }data;
    data.bytes[3] = (val>>(8*3))& 0xff;
    data.bytes[2] = (val>>(8*2))& 0xff;
    data.bytes[1] = (val>>(8*1))& 0xff;
    data.bytes[0] = (val>>(8*0))& 0xff;


    return data.val_float;
}
*/


void initPeripherals()
{
	//Initialize im2gemm core
	printf("Initializing fp_im2gemm\n");
	im2gemm_cfg = XFp_im2gemm_LookupConfig(XPAR_FP_IM2GEMM_0_DEVICE_ID);
	if(im2gemm_cfg)
	{
		int status = XFp_im2gemm_CfgInitialize(&im2gemm, im2gemm_cfg);
		if(status != XST_SUCCESS)
		{
			printf("Error initializing im2gemm core\n");
		}
	}

	//initialize dma core
	printf("initializing DMA core\n");
	axiDMA_cfg = XAxiDma_LookupConfig(XPAR_AXIDMA_0_DEVICE_ID);
	if(axiDMA_cfg)
	{
		int status = XAxiDma_CfgInitialize(&axiDMA, axiDMA_cfg);
		if(status != XST_SUCCESS)
		{
			printf("Error initializing DMA core\n");
		}
	}

	XAxiDma_IntrDisable(&axiDMA, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&axiDMA, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);
}

int main()
{
	AxiTimerHelper Mytimer;
	initPeripherals();

	fp *hw_out = (fp*)malloc(sizeof(fp)*SIZE_ARR);

    int width = 224;
    int height = 224;
    int channels = 1;
    int ksize = 3;
    int stride = 1;
    int pad = 1;
    int filter = 1;

    int input_size = width*height*channels;
    int weight_size = ksize*ksize*channels;
    int out_w = (width + 2*pad - ksize)/stride + 1;
    int out_h = (height + 2*pad - ksize)/stride + 1;


    float *in_img = (float*)malloc(width*height*channels*sizeof(float));
    float *w_buf = (float*)malloc(filter*ksize*ksize*channels*sizeof(float));
    float *im2col_img = (float*)malloc(width*height*channels*9*sizeof(float));
    float *sw_out = (float*)malloc(filter*width*height*sizeof(float));
    //float *hw_out = (float*)malloc(filter*width*height*sizeof(float));


    for(int i =0; i<width*height*channels; i++){
        in_img[i] = INPUT[i];
    }
    for(int i =0; i<filter*ksize*ksize*channels; i++){
        w_buf[i] = WEIGHT[i];
    }


    for(int i =0; i<filter*width*height;i++){
        sw_out[i] = 0;
    }


    Mytimer.startTimer();
        if(ksize==3){
            im2col_cpu(in_img, channels, height, width, ksize, stride, pad, im2col_img);
            gemm_nn(filter, width*height, ksize*ksize*channels, 1.0, w_buf, ksize*ksize*channels, im2col_img, width*height, sw_out, width*height);

        }
        else gemm_nn(filter, width*height, ksize*ksize*channels, 1.0, w_buf, ksize*ksize*channels, in_img, width*height, sw_out, width*height);
       //scale_bias(sw_out, 0.5, 1, 1, width*height*filter);
       //add_bias(sw_out, 2.0, 1, 1, width*height*filter);

    Mytimer.stopTimer();
    float sw_time = Mytimer.getElapsedTimerInSeconds();
    //printf("SW complete : %f\n",sw_time);



	fp IN[SIZE_ARR];


	for(int i=0;i<weight_size+input_size;i++){//INPUT initialization
		if(i==input_size+weight_size) break;

		if(i<weight_size) {
			IN[i]=WEIGHT[i];
		}

		if(i<input_size+weight_size && i>=weight_size){
			IN[i]=INPUT[i-weight_size];
		}

	}


	Mytimer.startTimer();

	XFp_im2gemm_Set_height(&im2gemm,height);
	XFp_im2gemm_Set_width(&im2gemm,width);
	XFp_im2gemm_Set_ksize(&im2gemm,ksize);
	XFp_im2gemm_Set_channels(&im2gemm,channels);
	XFp_im2gemm_Set_stride(&im2gemm,stride);
	XFp_im2gemm_Set_pad(&im2gemm,pad);
	XFp_im2gemm_Set_out_reg(&im2gemm,1);
	XFp_im2gemm_Set_init_reg(&im2gemm,1);
	XFp_im2gemm_Set_in_reg(&im2gemm,1);
	XFp_im2gemm_Set_w_reg(&im2gemm,1);
	XFp_im2gemm_Set_out_index(&im2gemm,1);

	XFp_im2gemm_Start(&im2gemm);

	Xil_DCacheFlushRange((u32)IN, SIZE_ARR*sizeof(fp));
	Xil_DCacheFlushRange((u32)hw_out, SIZE_ARR*sizeof(fp));

	//printf("Sending data\n");
	XAxiDma_SimpleTransfer(&axiDMA, (u32)IN, SIZE_ARR*sizeof(fp),XAXIDMA_DMA_TO_DEVICE);
	//while(XAxiDma_Busy(&axiDMA, XAXIDMA_DEVICE_TO_DMA));

	XAxiDma_SimpleTransfer(&axiDMA, (u32)hw_out, SIZE_ARR*sizeof(fp),XAXIDMA_DEVICE_TO_DMA);
	//printf("Getting data\n");

	while(XAxiDma_Busy(&axiDMA, XAXIDMA_DEVICE_TO_DMA));
	//printf("DMA is not busy\n");

	Xil_DCacheFlushRange((u32)hw_out,sizeof(fp)*SIZE_ARR);

	while(!XFp_im2gemm_IsDone(&im2gemm));
	Mytimer.stopTimer();

	for(int i=0;i<out_h*out_w;i++){
		printf("hw_out[%d] = %f\t sw_out[%d]=%f\n",i,(float)hw_out[i],i,sw_out[i]);
	}

	float hw_time = Mytimer.getElapsedTimerInSeconds();

	printf("SW complete : %f\n",sw_time);
	printf("HW complete : %f\n",hw_time);

	return 0;
}




