clc
clear all;
close all;
warning('off');
sf    =   4;
addpath(genpath('data\'));
addpath(genpath('GNLTN-main\'));
for data_num=2
clear img
switch data_num    
    case 1
         S=imread('.\data\original_rosis.tif');
         img=S(1:256,1:256,11:end);
        name= 'original_rosis';
    case 2
        load 'Pavia_80.mat'
         img=OriData3;
         name='Pavia_80';
end  
img=double(img);
O_Img = img/max(img(:));
S= O_Img ;
[M N L]=size(S);
sz=[M N];
%%
load('.\data\R.mat');
F=R;
% F=F(:,1:end-10);
F=F(:,1:end-23);
for band = 1:size(F,1)
        div = sum(F(band,:));
        for i = 1:size(F,2)
            F(band,i) = F(band,i)/div;
        end
end
%%
Comparison_GNLTN = 1;
s0=1;
downsampling_scale = sf;
psf        =    fspecial('gaussian',7,2);
par.fft_B      =    psf2otf(psf,sz);
par.fft_BT     =    conj(par.fft_B);
par.H          =    @(z)H_z(z, par.fft_B, sf, sz,s0 );
par.HT         =    @(y)HT_y(y, par.fft_BT, sf, sz,s0);
[M,N,L] = size(S);
S_bar = hyperConvert2D(S);
hyper= par.H(S_bar);
SNRh=30;
sigma = sqrt(sum(hyper(:).^2)/(10^(SNRh/10))/numel(hyper));
rng(10,'twister')
hyper = hyper+ sigma*randn(size(hyper));
HSI=hyperConvert3D(hyper,M/downsampling_scale, N/downsampling_scale );
rng(10,'twister')
Y = F*S_bar;
SNRm=35;
sigmam = sqrt(sum(Y(:).^2)/(10^(SNRm/10))/numel(Y));
Y = Y+ sigmam*randn(size(Y));
MSI=hyperConvert3D(Y,M,N);
%% 15
if Comparison_GNLTN== 1;
 ex15='GNTNN';
cd 'GNLTN-main'
addpath(genpath(cd));
aa = 0;
para.K=400;
para.eta=[0.01, 0.01, 0.05];
para.p=10;
t0=clock;
Z15 =  Gernalized_GNLTN(HSI,MSI,F, par.fft_B,sf,S,para,aa); %% V0
t15=etime(clock,t0);    

[psnr15,rmse15, ergas15, sam15, uiqi15,ssim15,DD15,CC15] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z15)), 0, 1.0/sf);
disp(['PSNR_',ex15,'_',name ' = ',num2str(psnr15)]);
disp(['SSIM_',ex15,'_',name ' = ',num2str(ssim15)]);
disp(['ergas_',ex15,'_',name ' = ',num2str(ergas15)]);
disp(['sam_',ex15,'_',name ' = ',num2str(sam15)]);
disp(['uiqi_',ex15,'_',name ' = ',num2str(uiqi15)]);
disp(['DD_',ex15,'_',name ' = ',num2str(DD15)]);
disp(['CC_',ex15,'_',name ' = ',num2str(CC15)]);
aa = aa +1;
    end  
end