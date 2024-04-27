clear all;clc;close all
hdr_dataset='D:\HDR dataset\HDR4RTT Database\HDR4RTT Database\images';

hdrimagesPath=fullfile(hdr_dataset,'*.exr');
hdrimages=dir(hdrimagesPath);

for k=1 :length(hdrimages)
   
    image_name=hdrimages(k).name;
    imagePath=fullfile(hdr_dataset,image_name);
    img=hdrimread(imagePath);
     img = max(img, 0);
  [H,W,C]=size(img);
 outimages=zeros(14,H,W,3);
 outimages(1,:,:,:)=AshikhminTMO(img,0.6, true);
 
 [outimages(2,:,:,:),seg,BTMO]=BanterleTMO(img,4, 0.6);

 [outimages(3,:,:,:),exposure_val]=BestExposureTMO(img);
 
 outimages(4,:,:,:)=BruceExpoBlendTMO(img);

 outimages(5,:,:,:)=ChiuTMO(img);
 
 outimages(6,:,:,:)=DragoTMO(img);

 outimages(7,:,:,:)=ExponentialTMO(img,0.8);

 outimages(8,:,:,:)=LischinskiTMO(img);

 outimages(9,:,:,:)=LogarithmicTMO(img);

 outimages(10,:,:,:)=PattanaikTMO(img);

 [outimages(11,:,:,:),palpha,lwa]=ReinhardRobustTMO(img);
 
 [outimages(12,:,:,:),pAlpha,pWhite]=ReinhardTMO(img);
 outimages(13,:,:,:)=ATT_TMO(img);
 %%%%%%%%%%%%%%%%%%%%%%%L1-T10-Tone-mapping%%%%%%%%%%%%%%%%%%%%%%%%
 hdr_h = rgb2hsv(img);
 [hei,wid,channel] = size(img);
 hdr_l = hdr_h(:,:,3);
 hdr_l = log(hdr_l+0.0001);
 hdr_l = nor(hdr_l);
 lambda1 = 0.3;  
 lambda2 = lambda1*0.01;
 lambda3 = 0.1;
 %  decomposition
[D1,D2,B2] = Layer_decomp(hdr_l,lambda1,lambda2,lambda3);

% Scaling
sigma_D1 = max(D1(:));
D1s = R_func(D1,0,sigma_D1,0.8,1);

B2_n= compress(B2,2.2,1);
hdr_lnn = 0.8*B2_n + D2 + 1.2*D1s;
% postprocessing
hdr_lnn = nor(clampp(hdr_lnn,0.005,0.995));
outimages(14,:,:,:) = hsv2rgb((cat(3,hdr_h(:,:,1),hdr_h(:,:,2)*0.6,hdr_lnn)));
 

 % YPTumblinTMO(img);
 % YPWardGlobalTMO(img);
 [imgout,tmqi]=BestTMQI(img,outimages);
 imgName=[RemoveExt(image_name) '.png'];
 newFileName=fullfile('D:\HDR dataset\HDR4RTT Database\HDR4RTT Database\ldr',imgName);
 imwrite(uint8(imgout),newFileName);
 tmqi
   end
  
function [imgout,tmqi]=BestTMQI(img,ldr)
tmqi=zeros(14,1);
for i=1:14
    tmqi(i,1)=TMQI(img,squeeze(ldr(i,:,:,:))*255);
end
[a,d]=max(tmqi);
imgout=squeeze(ldr(d,:,:,:)*255);
end
