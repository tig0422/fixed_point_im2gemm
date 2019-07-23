#include <hls_stream.h>
#include <stdio.h>
#include <stdlib.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
//#define T float


#define PE 16

//16bit total size, 9bit integer bit(1bit = sign bit), 7bit fractional
typedef ap_fixed<16,9> fp;

struct data_struct {
    fp data;
    bool last;
};

fp im2col_get_pixel(fp *im, int height, int width, int channels, int row, int col, int channel, int pad) {
#pragma HLS INLINE
    row -= pad;//correction as pad
    col -= pad;//correction as pad
    if (row < 0 || col < 0 || row >= height || col >= width) return 0;//padding part = 0
    return im[col + width*row + width*height*channel];
}


void im2mul(fp *data_im, int channels,  int height,  int width, int ksize,  int stride,
		int pad, fp *data_col, int k_offset, fp *WEIGHT, int out_index)
{
    int height_col = (height + 2*pad - ksize) / stride + 1; //feature map size
    int width_col = (width + 2*pad - ksize) / stride + 1; //feature map size
    int channels_col = channels * ksize * ksize; //total kernel size

    fp im2_buf[PE];
#pragma HLS ARRAY_PARTITION variable=im2_buf complete dim=1
    fp out_buf[PE];
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=1

/*
    fp data_im_buf[50176];

#pragma HLS RESOURCE variable=data_im_buf core=RAM_2P_BRAM

    for(int i=0;i<50176;i++){
    	if(i==height*width*channels) break;
#pragma HLS PIPELINE II=1
    	data_im_buf[i]=data_im[i];
    }
*/

    for(int i =0; i<PE; i++){//Initialize out_buf
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL factor = 16
    	out_buf[i] = 0;
    }

	float w_reg;
	int height_end = 0;
	//height_end = 0;
	int temp_reg;
	//int count=0;
	int c_temp=-1;
	int w_offset;
	int h_offset;
	int c_im;
	int im_col;
	int w_offset_table[9] = {0,1,2,0,1,2,0,1,2};
	int h_offset_table[9] = {0,0,0,1,1,1,2,2,2};
	int c_im_table[9] = {0,0,0,0,0,0,0,0,0};


WIDTH_LOOP:
	for (int w = 0; w < 224; ++w) {//input width size
		if(w==width_col) break;//max feature map width = 224
      HEIGHT_LOOP:
		   for (int h = 0; h < 224/PE; ++h) { //input height size, PE=16 -> h<14
          CHANNEL_LOOP:
				for (int c = 0; c < 576; ++c) {//max weight = 32
#pragma HLS PIPELINE II = 1
					if(c==channels_col) break;// repeat as total weight size
/************************NEW PARAMETER************************************/

					if(ksize==3){
						c_temp++;
						w_offset = w_offset_table[c_temp];
						h_offset = h_offset_table[c_temp];

						im_col = w_offset + w * stride;
						c_im = c_im_table[c_temp];
						c_im_table[c_temp]++;
/*
						printf("---------c = %d----------\n",c);
						printf("c_temp = %d\n",c_temp);
						printf("w_offset = %d\n",w_offset);
						printf("h_offset = %d\n",h_offset);
						printf("c_im = %d\n",c_im);
*/
						if(c_temp == 8) {
							if(channels==2) for(int i=0 ; i<9 ; i++) {// compute two channels in one iteration
							c_im_table[i]=0;
							}
						if(c==channels_col-1) for(int i=0 ; i<9 ; i++) {
							c_im_table[i]=0;
							}
						//printf("c_im_table = %d %d %d %d %d %d %d %d %d\n\n",c_im_table[0],c_im_table[1],c_im_table[2],c_im_table[3],c_im_table[4],c_im_table[5],c_im_table[6],c_im_table[7],c_im_table[8]);
						c_temp = -1;
						}


					}
					else{
						w_offset=0;
						h_offset=0;
						c_im=c;
						im_col = w_offset + w*stride;//
					}

/****************************OLD PARAMETER*****************************************/

/*

					int w_offset = c%ksize;//012012012 012012012 012012012

					int h_offset = (c / ksize) % ksize;//000111222 000111222 000111222 -> 3. row repeat

					int c_im = c / ksize / ksize;	   //000000000 111111111 222222222 -> 1. go to next input channel

					int im_col = w_offset + w * stride;//012012012 012012012 012012012 -> 2. column repeat when w=0
													   //123123123 123123123 123123123 -> when w=1, stride=1
*/

				IM2COL_LOOP://LOOP under PIPELINE will be unrolled automatically
					for(int i =0; i<PE;i++){
						if(PE*h+i<height_col) {
//#pragma HLS UNROLL factor = 16
							im2_buf[i] = im2col_get_pixel(data_im, height, width, channels, h_offset + (PE*h+i) * stride, im_col, c_im, pad);
						} else  height_end = 1;
					}

		         MAC_LOOP:
					for(int i =0; i<PE; i++){
//#pragma HLS UNROLL factor = 16
						if((PE*h+i)<height_col) out_buf[i] += im2_buf[i]*WEIGHT[c];//accumulate im2_buf*weight to the out_buf
					}

                 }

	WRITE_LOOP:	for(int i =0; i<PE;i++){
					if((PE*h+i)<height_col){
						data_col[width_col*(PE*h+i)+w] += out_buf[i];
						out_buf[i] = 0;
					}
				}

                if(height_end) {
                	height_end = 0;
                    break;
                }
            }
        }
}


void fp_im2gemm(hls::stream<fp> &INPUT_STREAM, hls::stream<data_struct> &OUTPUT_STREAM, int height, int width, int channels, int ksize, int stride, int pad,
		int out_reg, int init_reg, int in_reg, int w_reg, int out_index){

#pragma HLS INTERFACE s_axilite port=height
#pragma HLS INTERFACE s_axilite port=width
#pragma HLS INTERFACE s_axilite port=return     bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=channels
#pragma HLS INTERFACE s_axilite port=ksize
#pragma HLS INTERFACE s_axilite port=stride
#pragma HLS INTERFACE s_axilite port=pad
#pragma HLS INTERFACE s_axilite port=out_reg
#pragma HLS INTERFACE s_axilite port=in_reg
#pragma HLS INTERFACE s_axilite port=w_reg
#pragma HLS INTERFACE s_axilite port=init_reg
#pragma HLS INTERFACE s_axilite port=out_index
#pragma HLS INTERFACE axis port=OUTPUT_STREAM
#pragma HLS INTERFACE axis port=INPUT_STREAM


    fp IN[100352];
    fp WEIGHT[576];
    fp OUT[50176];


    int in_size = height*width*channels;
    int weight_size = ksize*ksize*channels;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;

    unsigned short output_size = height_col*width_col;

    data_struct out_data;
/*
    float reg_index;
    float reg_value1;
    float reg_value2;
    unsigned int temp;
    unsigned short temp1;
    unsigned short temp2;

    union _data_temp{
        unsigned int temp;
        unsigned short temp_arr[2];
    }data_temp;
    unsigned short index_reg_num;
    unsigned short trans=0;
    trans = 0;
*/

    if(init_reg){
    	for(int i =0; i<50176; i++){
			#pragma HLS PIPELINE II=1
    		OUT[i] = 0;
    	}
    }


	if(w_reg){
	WEIGHT_STREAM_LOOP:
		for(int i =0; i<576; i++){
			if(i==weight_size) break;
	#pragma HLS PIPELINE II=1
			WEIGHT[i] = INPUT_STREAM.read();
			//printf("w[%d] : %f\n",i,(float)WEIGHT[i]);
		}
	}


    if(in_reg){
    	//if(control_comp==0){
			INPUT_STREAM_LOOP:
			for(int i =0; i<100352; i++){
				if(i==in_size) break;
				#pragma HLS PIPELINE II=1
				//#pragma HLS unroll factor = 2
				IN[i] = INPUT_STREAM.read();
				//if(init) OUT[i] = 0;
			}

    }

	im2mul(IN, channels, height, width, ksize,  stride, pad, OUT, 0, WEIGHT, out_index);

	// core end
	if(out_reg){
	OUTPUT_STREAM_LOOP:
		for(int i = 0; i<50176; i++){
			if(i==(output_size)) break;
		#pragma HLS PIPELINE II=1
			out_data.data = OUT[i];
			out_data.last = (i==(output_size-1))?1:0;
			OUTPUT_STREAM.write(out_data);
		}
	}
}
