function  [Z] =  Gernalized_GNLTN( HSI, MSI,R,FBm,sf,S,para,aaa)
K=para.K;
eta=para.eta(1);
eta2=para.eta(2);
eta3=para.eta(3);
inner = 5;
p=para.p;
mu1=1e-2;
mu2=1e-2;
mu3=1e-2;
mu4=1e-3;
patchsize=6;
overlap=4;
iii = 1;
HSI3=Unfold(HSI,size(HSI),3);
[D,~]=svds(HSI3,p);
RD=R*D;
L1=size(D,2);
[~,~,D2]=svds(HSI3,1);
D2=reshape(D2,[size(HSI,1),size(HSI,2)]);
D2=D2/(max(D2(:))-min(D2(:)))-min(D2(:))/(max(D2(:))-min(D2(:)));
D2=imresize(D2,sf);
path='E:\2024实验matlab\code2025\';
filename = [path 'mu4_1.xls']; 
bparams.block_sz = [patchsize, patchsize];
bparams.overlap_sz=[overlap overlap];
[nr, nc,~]=size(MSI);
L=size(HSI,3);
num1=(nr-patchsize)/(patchsize-overlap)+1;
num2=(nc-patchsize)/(patchsize-overlap)+1;
bparams.block_num=[num1 num2]; 
sizeag=size(MSI,3);
MSIG(:,:,1:sizeag)=MSI;
MSIG(:,:,sizeag+1)=D2;
predenoised_blocks = ExtractBlocks(MSIG, bparams);
Y2=Unfold(predenoised_blocks,size(predenoised_blocks),4);
if K==1
    aa=ones(num1*num2,1);
else
    [aa ]=fkmeans(Y2,K);
end
HSI_int=zeros(nr,nc,L);
HSI_int(1:sf:end,1:sf:end,:)=HSI;
FBmC  = conj(FBm);
FBs  = repmat(FBm,[1 1 L]);
FBs1  = repmat(FBm,[1 1 L1]);
FBCs=repmat(FBmC,[1 1 L]);
FBCs1=repmat(FBmC,[1 1 L1]);
HHH=ifft2((fft2(HSI_int).*FBCs));
HHH1=hyperConvert2D(HHH);
%% iteration
MSI3=Unfold(MSI,size(MSI),3);
n_dr=nr/sf;
n_dc=nc/sf;
HR_load1=imresize(HSI, sf,'bicubic');
V1=D'*hyperConvert2D(HR_load1);
V2=V1;
V3=V1;
P1=tanh_my(V1);
P2=tanh_my(V2); 
P3=tanh_my(V3); 
G1=zeros(size(V2));
G2=G1;
G3=G1;
M1 = G1;
M2 = G1;
M3 = G1;
DTD=D'*D;
CCC=DTD\(RD'*MSI3+D'*HHH1);
C1=DTD\(RD'*RD+3*mu1*eye(size(D,2))); 
[Q,Lambda]=eig(C1);
Lambda=reshape(diag(Lambda),[1 1 L1]);
InvLbd=1./repmat(Lambda,[ sf*n_dr  sf*n_dc 1]);
B2Sum=PPlus(abs(FBs1).^2./( sf^2),n_dr,n_dc);
InvDI=1./(B2Sum(1:n_dr,1:n_dc,:)+repmat(Lambda,[n_dr n_dc 1]));
for i=1:50
HR_HSI3=mu1*(V1+G1/(2*mu1)+V2+G2/(2*mu1)+V3+G3/(2*mu1));
C3=CCC+DTD\HR_HSI3;
C30=fft2(reshape((Q\C3)',[nr nc L1])).*InvLbd;
temp  = PPlus_s(C30/( sf^2).*FBs1,n_dr,n_dc);
invQUF = C30-repmat(temp.*InvDI,[ sf  sf 1]).*FBCs1; % The operation: C5bar- temp*(\lambda_j d Im+\Sum_i=1^d Di^2)^{-1}Dv^H)
VXF    = Q*reshape(invQUF,[nc*nc L1])';
A = reshape(real(ifft2(reshape(VXF',[nr nc L1   ]))),[nc*nc L1])'; 
Zt=hyperConvert3D(D*A,nr, nc );
psnr(i)=csnr(double(im2uint8(S)),double(im2uint8(Zt)),0,0)
%%
temp11  = A-G1/(2*mu1);
temp12  =P1+M1/(2*mu2);
V1  = Newton(temp12,temp11,V1,inner,(2*mu1),(2*mu2));

temp21  = A-G2/(2*mu1);
temp22  =P2+M2/(2*mu3);
V2  = Newton(temp22,temp21,V2,inner,(2*mu1),(2*mu3));

temp31  = A-G3/(2*mu1);
temp32  =P3+M3/(2*mu4);
V3  = Newton(temp32,temp31,V3,inner,(2*mu1),(2*mu4));
%% spatial similarties
B1=hyperConvert3D(tanh_my(V1)-M1/(2*mu2),nr, nc );
predenoised_blocks1 = ExtractBlocks(B1, bparams);
Z_block1=zeros( bparams.block_sz(1), bparams.block_sz(2),L1, bparams.block_num(1)* bparams.block_num(2));
B2=hyperConvert3D(tanh_my(V2)-M2/(2*mu3),nr, nc );
predenoised_blocks2 = ExtractBlocks(B2, bparams);
Z_block2=zeros( bparams.block_sz(1), bparams.block_sz(2),L1, bparams.block_num(1)* bparams.block_num(2));
B3=hyperConvert3D(tanh_my(V3)-M3/(2*mu4),nr, nc );
predenoised_blocks3 = ExtractBlocks(B3, bparams);
Z_block3=zeros( bparams.block_sz(1), bparams.block_sz(2),L1, bparams.block_num(1)* bparams.block_num(2));


for mn=1:max(aa)
gg=find(aa==mn);
XES=predenoised_blocks1(:,:,:,gg);  
[a, b, c, d ]=size(XES);
XES = reshape(XES,[a*b c d]);
P1=Log_prox_tnn_FL(XES, eta/2/mu2);
P1=reshape(P1,[a b c d]); 
Z_block1(:,:,:,gg)=P1;

XES=predenoised_blocks2(:,:,:,gg);
[a, b, c, d ]=size(XES);
XES = reshape(XES,[a*b c d]);
XES=permute(XES,[1,3,2]);
P2=Log_prox_tnn_FL(XES, eta2/2/mu3);
P2=permute(P2,[1,3,2]);
P2=reshape(P2,[a b c d]); 
Z_block2(:,:,:,gg)=P2;
 
XES=predenoised_blocks3(:,:,:,gg);
[a, b, c, d ]=size(XES);
XES = reshape(XES,[a*b c d]);
XES=permute(XES,[2,3,1]);
P3=Log_prox_tnn_FL(XES, eta3/2/mu4);
P3=permute(P3,[3,1,2]);
P3=reshape(P3,[a b c d]); 
Z_block3(:,:,:,gg)=P3;
  
end
    
P1= JointBlocks(Z_block1, bparams);
P1=hyperConvert2D(P1);
G1=G1+2*mu1*(V1-A);
M1 = M1 + 2*mu2*(P1-tanh_my(V1));
P2= JointBlocks(Z_block2, bparams);
P2=hyperConvert2D(P2);
G2=G2+2*mu1*(V2-A);
M2 = M2 + 2*mu3*(P2-tanh_my(V2)); 
P3= JointBlocks(Z_block3, bparams);
P3=hyperConvert2D(P3);
G3=G3+2*mu1*(V3-A);
M3 = M3 + 2*mu4*(P3-tanh_my(V3));
end
Z=hyperConvert3D(D*A,nr, nc );
end

function Z  = Newton(g,a,Z,inner,alpha, beta)
    i = 0;
    relchg = 1;
    tol = 10^(-4);  
    while i < inner && relchg > tol
        Zp = Z;
        
        % 计算 Numer 和 Denom
        Numer = beta .* (1 - tanh_my(Z).^2) .* (tanh_my(Z) - g) + alpha .* (Z - a);
        Denom = -2 .* beta .* tanh_my(Z) .* (1 - tanh_my(Z).^2) .* (tanh_my(Z) - g) + beta .* (1 - tanh_my(Z).^2).^2 + alpha;
        % 更新 Z
        Z = Z - Numer ./ Denom;
        
        % 检查 Z 的最大值是否超过 50
        if max(Z(:)) > 50 || min(Z(:)) < -500
            Z = Zp;
            return;  % 退出函数
        end
        
        % 计算相对变化
        relchg = norm(Z - Zp, 'fro') / norm(Z, 'fro');
        
        % 增加迭代次数
        i = i + 1;
    end
end


   function output = tanh_my(x)
      output =(exp(x) - exp(-x))./(exp(x) + exp(-x));
   end
 



