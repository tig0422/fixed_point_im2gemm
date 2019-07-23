            if(CONV==1){
                FILE *fp=NULL;
                fp=fopen("input.h","w");
                fprintf(fp,"float INPUT[%d] = {",l.w * l.h * l.c);
                for(int i=0;i<l.w*l.h*l.c;i++){
                    if(i<l.w*l.h*l.c-1)fprintf(fp,"%f,\n",im[i]);
                    else fprintf(fp,"%f\n",im[i]);
                }
                fprintf(fp,"}");
                
                
                fp=fopen("weight.h","w");
                fprintf(fp,"float WEIGHT[%d] = {",l.size * l.size * l.c * m);
                for(int i=0;i<l.size*l.size*l.c*m;i++){
                    if(i<l.size*l.size*l.c*m-1)fprintf(fp,"%f,\n",a[i]);
                    else fprintf(fp,"%f\n",a[i]);
                }
                fprintf(fp,"}");                
                
                
                fp=fopen("output.h","w");
                fprintf(fp,"float OUTPUT[%d] = {",n * m);
                for(int i=0;i<n*m;i++){
                    if(i<n*m-1)fprintf(fp,"%f,\n",c[i]);
                    else fprintf(fp,"%f\n",c[i]);
                }
                fprintf(fp,"}");
                
            }
