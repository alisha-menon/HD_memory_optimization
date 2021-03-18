clear;

%====Features and Label===
load('input_data.mat')
features=data_all;
subjects = 33;
repetitions = 10;
f_label_a_binary=data_all(:,215);
f_label_v_binary=data_all(:,216);

for k=1:214
features(:,k)=features(:,k)-min(features(:,k));
end

for k=1:214
 features(:,k)=features(:,k)/max(features(:,k));
end

for i=1:214
 features(:,i)=features(:,i)-0.4;
end

features_GSR=features(:,1:32);
features_ECG=features(:,1+32:32+77); 
features_EEG=features(:,1+32+77:32+77+105); 

%=======HDC============
HD_functions;     % load HD functions
learningrate=0.25;% percentage of the dataset used to train the algorithm
acc_ngram_1=[];
acc_ngram_2=[];
accuracy_A = zeros(subjects,repetitions);
accuracy_V = zeros(subjects,repetitions);


channels_v=length(features_GSR(1,:));
channels_v_ECG=length(features_ECG(1,:));
channels_v_EEG=length(features_EEG(1,:));

channels_a=channels_v;
channels_a_ECG=channels_v_ECG;
channels_a_EEG=channels_v_EEG;

COMPLETE_1_v=features_GSR;
COMPLETE_1_a=features_GSR;
COMPLETE_1_v_ECG=features_ECG;
COMPLETE_1_a_ECG=features_ECG;
COMPLETE_1_v_EEG=features_EEG;
COMPLETE_1_a_EEG=features_EEG;

for j=1:repetitions
learningFrac = learningrate(1);
D = 10000; %dimension of the hypervectors
classes = 2; % level of classes
precision = 20; %no use
ngram = 4; % for temporal encode
maxL = 2; % for IM gen
 
channels_v_EXG=channels_v +channels_v_ECG+channels_v_EEG;
channels_a_EXG=channels_a+channels_a_ECG+channels_a_EEG;


[chAM1, iMch1] = initItemMemories (D, maxL, channels_v);
[chAM2, iMch2] = initItemMemories (D, maxL, channels_a);
[chAM3, iMch3] = initItemMemories (D, maxL, channels_v_ECG);
[chAM4, iMch4] = initItemMemories (D, maxL, channels_a_ECG);
[chAM5, iMch5] = initItemMemories (D, maxL, channels_v_EEG);
[chAM6, iMch6] = initItemMemories (D, maxL, channels_a_EEG);
[chAM7, iMch7] = initItemMemories (D, maxL, channels_v_EXG);
[chAM8, iMch8] = initItemMemories (D, maxL, channels_a_EXG);

%downsample the dataset using the value contained in the variable "downSampRate"
downSampRate = 8;
LABEL_1_v=f_label_v_binary;
LABEL_1_a=f_label_a_binary;
[TS_COMPLETE_1, L_TS_COMPLETE_1] = downSampling (COMPLETE_1_v, LABEL_1_v, downSampRate);
[TS_COMPLETE_2, L_TS_COMPLETE_2] = downSampling (COMPLETE_1_a, LABEL_1_a, downSampRate);
[TS_COMPLETE_3, L_TS_COMPLETE_3] = downSampling (COMPLETE_1_v_ECG, LABEL_1_v, downSampRate);
[TS_COMPLETE_4, L_TS_COMPLETE_4] = downSampling (COMPLETE_1_a_ECG, LABEL_1_a, downSampRate);
[TS_COMPLETE_5, L_TS_COMPLETE_5] = downSampling (COMPLETE_1_v_EEG, LABEL_1_v, downSampRate);
[TS_COMPLETE_6, L_TS_COMPLETE_6] = downSampling (COMPLETE_1_a_EEG, LABEL_1_a, downSampRate);

%generate the training matrices using the learning rate contined in the
%variable "learningFrac"

for subject = 1:subjects
subject

[L1_1, L2_1, L_SAMPL_DATA_1, SAMPL_DATA_1, L_SAMPL_DATA_1_test, SAMPL_DATA_1_test] = genTrainTestData (TS_COMPLETE_1, L_TS_COMPLETE_1, learningFrac, 'inorder', subject);
[L1_2, L2_2, L_SAMPL_DATA_2, SAMPL_DATA_2, L_SAMPL_DATA_2_test, SAMPL_DATA_2_test] = genTrainTestData (TS_COMPLETE_2, L_TS_COMPLETE_2, learningFrac, 'inorder', subject);
[SAMPL_DATA_3, SAMPL_DATA_3_test] = select_traintest(L1_1, L2_1, TS_COMPLETE_3);
[SAMPL_DATA_4, SAMPL_DATA_4_test] = select_traintest(L1_2, L2_2, TS_COMPLETE_4);
[SAMPL_DATA_5, SAMPL_DATA_5_test] = select_traintest(L1_1, L2_1, TS_COMPLETE_5);
[SAMPL_DATA_6, SAMPL_DATA_6_test] = select_traintest(L1_2, L2_2, TS_COMPLETE_6);

%Sparse biopolar mapping
q=0.7;
projM1=projBRandomHV(D,channels_v,q);
projM2=projBRandomHV(D,channels_a,q);
projM3=projBRandomHV(D,channels_v_ECG,q);
projM4=projBRandomHV(D,channels_a_ECG,q);
projM5=projBRandomHV(D,channels_v_EEG,q);
projM6=projBRandomHV(D,channels_a_EEG,q);


%for N = 1 : ngram
for N = 4 : ngram
N

fprintf ('HDC for Arousal\n');
[numpat_2, hdc_model_2] = hdctrainproj (L_SAMPL_DATA_2, SAMPL_DATA_2, chAM8, iMch2, D, N, precision, channels_a,projM2); 
[numpat_4, hdc_model_4] = hdctrainproj (L_SAMPL_DATA_2, SAMPL_DATA_4, chAM8, iMch4, D, N, precision, channels_a_ECG,projM4); 
[numpat_6, hdc_model_6] = hdctrainproj (L_SAMPL_DATA_2, SAMPL_DATA_6, chAM8, iMch6, D, N, precision, channels_a_EEG,projM6); 

hdc_model_2(1)=mode([hdc_model_2(1); hdc_model_4(1); hdc_model_6(1)]);
hdc_model_2(2)=mode([hdc_model_2(2); hdc_model_4(2); hdc_model_6(2)]);

fprintf ('HDC for Valence\n');
[numpat_1, hdc_model_1] = hdctrainproj (L_SAMPL_DATA_1, SAMPL_DATA_1, chAM8, iMch1, D, N, precision, channels_a,projM1); 
[numpat_3, hdc_model_3] = hdctrainproj (L_SAMPL_DATA_1, SAMPL_DATA_3, chAM8, iMch3, D, N, precision, channels_a_ECG,projM3); 
[numpat_5, hdc_model_5] = hdctrainproj (L_SAMPL_DATA_1, SAMPL_DATA_5, chAM8, iMch5, D, N, precision, channels_a_EEG,projM5); 

hdc_model_1(1)=mode([hdc_model_1(1); hdc_model_3(1); hdc_model_5(1)]);
hdc_model_1(2)=mode([hdc_model_1(2); hdc_model_3(2); hdc_model_5(2)]);


for i=1:channels_a
iMch8(i)=iMch2(i);
end
for i=channels_a+1:channels_a+channels_a_ECG
iMch8(i)=iMch4(i-channels_a);
end
for i=channels_a+channels_a_ECG+1:channels_a+channels_a_ECG+channels_a_EEG
iMch8(i)=iMch6(i-channels_a-channels_a_ECG);
end


[acc_ex2, acc2, pl2, al2] = hdcpredictproj  (L_SAMPL_DATA_2_test, SAMPL_DATA_2_test, L_SAMPL_DATA_2_test, SAMPL_DATA_4_test, L_SAMPL_DATA_2_test, SAMPL_DATA_6_test, hdc_model_2, chAM8, iMch8, D, N, precision, classes, channels_a,channels_a_ECG,channels_a_EEG,projM2,projM4,projM6);
accuracy_A(subject,j) = acc2;
acc2

[acc_ex1, acc1, pl1, al1] = hdcpredictproj  (L_SAMPL_DATA_1_test, SAMPL_DATA_1_test, L_SAMPL_DATA_1_test, SAMPL_DATA_3_test, L_SAMPL_DATA_1_test, SAMPL_DATA_5_test, hdc_model_1, chAM8, iMch8, D, N, precision, classes, channels_v,channels_v_ECG,channels_v_EEG,projM1,projM3,projM5);
accuracy_V(subject,j) = acc1;
acc1

acc_ngram_1(N,j)=acc1;
acc_ngram_2(N,j)=acc2;
end

end
end

