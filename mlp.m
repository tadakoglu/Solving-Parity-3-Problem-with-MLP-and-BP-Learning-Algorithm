function [model, mse] = mlp(X, Y, h)
% Multilayer perceptron
% Input:
%   X: d x n data matrix %%  eðitim seti giriþ(input) kümesi
%   Y: p x n response matrix %% eðitim seti hedef(target) kümesi
%   h: L x 1 vector specify number of hidden nodes in each layer l %% her
%   bir katmandaki nöron sayýsý. mesela [2,3] , 1. gizli katmanda iki tane
%   ve 2. gizli katmanda 3 tane nöron bulunduðunu gösterir. 
%   Bunun dýþýndaki giriþ ve çýkýþ iþlemlerini yapan katmanlardaki nöronlarýn sayýsý giriþ ve target eðitim seti
%   kümesinden otomatik olarak ayarlanýr.
% Ouput:
%   model: model structure
%   mse: mean square error % ortalama kare hata.
h = [size(X,1);h(:);size(Y,1)]; %% burada görüldüðü üzere X giriþ vektörleri kümemizin satýr sayýsý yapay sinir aðý giriþi sayýsý otomatik olarak alýndý. ayný iþlem target(hedef) küme içinde yapýldý. gizli katmanlar ve bu katmanlardaki nöron sayýlarý h ile önceden belirlenmiþti.
L = numel(h); %% numel eleman sayýsýný hesaplar bu durumda L gizli katman sayýsý olacaktýr.
W = cell(L-1);
for l = 1:L-1
    W{l} = randn(h(l),h(l+1)); %% mesela ilk döngüde 1. katman nöron sayýsý x 2. katman nöron sayýsý w1 deðerini oluþturur. {} indisleme için kullanýlýr.
end
Z = cell(L); %% katman sayýsý x katmansayýsý büyüklüðünde boþ matrix
Z{1} = X;
eta = 1/size(X,2);
% eta=0.04;
maxiter = 20000; %% maximum iterasyon sayýsý belirledik

mse = zeros(1,maxiter);
for iter = 1:maxiter
%     forward %1. adým ileri yönlü yayýlým
    for l = 2:L
        Z{l} = HiperbolicTan(W{l-1}'*Z{l-1});
    end
%     backward %2. adým geri yönlü yayýlým
    E = Y-Z{L};
    mse(iter) = mean(dot(E(:),E(:))); % ortalama kare hesapladýk. burada LMS WidrofHoff kuralýnýn genelleþtirmiþ hali kullanýldý, backpropagation algorithm MSE..
    for l = L-1:-1:1
        df = Z{l+1}.*(1-Z{l+1});
        dG = df.*E;
        dW = Z{l}*dG';
        W{l} = W{l}+eta*dW;
        E = W{l}*dG;
    end
end
mse = mse(1:iter); %aproximate yani yaklaþýk MSE(ORTALAMA KARE HATA) deðeri alýndýðýndan k. iterasyonda uygulandýðýndaki oluþan gerçek hata deðeri ortalam hata deðeri olarak kabul edilir.
model.W = W;