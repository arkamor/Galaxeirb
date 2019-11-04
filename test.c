#include <stdio.h>
#include <stdlib.h>


int main(int argc, char **argv){
    FILE *fd = fopen("dubinski.tab","r");
    if(fd==NULL){
        printf("error opening file\n");
        exit(0);
    }
    float mas[82000];
    float px[82000];
    float py[82000];
    float pz[82000];
    float vx[82000];
    float vy[82000];
    float vz[82000];
    char tst[200];
    for(int j;j<1024;j++){
        fscanf(fd,"%f %f %f %f %f %f %f",&mas[j],&px[j],&py[j],&pz[j],&vx[j],&vy[j],&vz[j]);
        for(int i;i<75;i++) scanf(NULL,fd);
    }
    for (int i=0;i<1024;i++){
        printf("%.8f %.4f %.4f %.4f %.4f %.4f\n",mas[i],px[i],py[i],pz[i],vx[i],vy[i],vz[i]);
    }
    
    return 0;
}