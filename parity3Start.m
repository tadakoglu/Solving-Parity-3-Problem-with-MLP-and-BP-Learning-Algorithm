%TAYFUN ADAKOÐLU
%11253001
%PARITY 3 PROBLEMLEMÝ BACKPROPAGATION VE MULTILAYER PERCEPTRON ILE COZUMU ODEV 5
%Detaylý anlatýmým için bu projedeki mlp.m dosyasýný okuyunuz. 
%Saygýlarýmla

%EÐER YAPAY SÝNÝR AÐI YAKINSAMIYOR ÝSE TEKRAR ÇALIÞTIRINIZ
%FARKLI BAÞLANGIÇ PARAMETRE DEÐERLERÝ ATANACAKTIR ANCAK BU ÝÞLEME ÝHTÝYAÇ DUYULMA OLASILIÐI DÜÞÜKTÜR
%MÝNÝMUM NÖRON SAYILARINI SEÇMEYE ÇALIÞTIM.
%AÐ BAÞLANGIÇ NOKTALARININ ÇOÐUNDA(RASGELE DENENMÝÞ) ÇALIÞMAKTADIR TEST EDÝLMÝÞTR.
%ANCAK BAZEN 3. YENÝDEN BAÞLATMADA YAKINSAYABÝLMEKTEDÝR. KESÝN AÐIRLIK
%DEÐERLERÝ BU ÝÞLEMLERÝ GÖSTERMEK ÝÇÝN ATANMAMIÞTIR...

% PARITY 3 PROBLEMÝ 2 KATMNANLI VE  3(GÝRÝÞ)-3-1 ÞEKLÝNDE TOPLAM 4 NÖRONLU OR(2. katman) VE AND TRANSFER FONKSÝYONU ÝÇEREN
% PERCEPTRON ÝLE DE ÇÖZÜLEBÝLÝR.

clear; close all;
h = [25,10]; % gizli katmanlardaki nöron sayýlarý. 1. katmanda 1 adettir; 2. katmandaki nöron sayýsý virgül çekilip  ifade edilebilir


%%xor için (x,t) eðitim seti. test için kullanýlmýþtýr ödev ile alakasýz.
% X = [0 0 1 1;0 1 0 1];
% T = [0 1 1 0];  

% PARITY 3 problemi için (x,t) eðitim kümesi
X=transpose([-1 -1 -1;-1 -1 1;-1 1 -1;-1 1 1 ;1 -1 -1;1 -1 1;1 1 -1;1 1 1]); % TRANSPOSE ALARAK KOLAYLIK YARATTIK.	
T=[-1 1 1 -1 1 -1 -1 1];

%PARITY 4 problemi için (x,t) eðitim kümesi, 
%LÜTFEN DENEMEK ÝÇÝN YORUMLARI(%) KALDIRINIZ
% 
 

[model,mse] = mlp(X,T,h);
plot(mse);
disp(['T = [' num2str(T) ']']);
Y = mlpPred(model,X);
disp(['Y = [' num2str(Y) ']']);