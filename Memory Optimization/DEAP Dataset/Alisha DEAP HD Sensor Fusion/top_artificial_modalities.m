clear;

%====Features and Label===
load('input_data.mat')
features=data_all;
f_label_a_binary=data_all(:,215);
f_label_v_binary=data_all(:,216);
met_A_accuracy = zeros(1,8);
met_V_accuracy = zeros(1,8);
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
features_ECG1=features(:,1+32:32+37); 
features_ECG2=features(:,1+32+37:32+77); 
features_ECG=features(:,1+32:32+77); 
features_EEG1=features(:,1+32+77:32+77+26); 
features_EEG15=features(:,1+32+77:32+77+52); 

features_EEG2=features(:,1+32+77+52:32+77+52+26);
features_EEG25=features(:,1+32+77+52:32+77+105);

features_EEG=features(:,1+32+77:32+77+105); 


%% choose select
% select = 1 for early fusion
% select = 2 for late fusion
select = 1;
if (select == 1)
    HD_functions_mod_reduced;     % load HD functions
else 
    HD_functions_multiplex;
end
    
learningrate=0.25; % percentage of the dataset used to train the algorithm
acc_ngram_1=[];
acc_ngram_2=[];


channels_v=length(features_GSR(1,:));
channels_v_ECG1=length(features_ECG1(1,:));
channels_v_ECG2=length(features_ECG2(1,:));
channels_v_ECG=length(features_ECG(1,:));
channels_v_EEG1=length(features_EEG1(1,:));
channels_v_EEG15=length(features_EEG15(1,:));
channels_v_EEG2=length(features_EEG2(1,:));
channels_v_EEG25=length(features_EEG25(1,:));
channels_v_EEG=length(features_EEG(1,:));

channels_a=channels_v;
channels_a_ECG1=channels_v_ECG1;
channels_a_ECG2=channels_v_ECG2;
channels_a_ECG =channels_v_ECG;
channels_a_EEG1=channels_v_EEG1;
channels_a_EEG15=channels_v_EEG15;
channels_a_EEG2=channels_v_EEG2;
channels_a_EEG25=channels_v_EEG25;
channels_a_EEG=channels_v_EEG;

COMPLETE_1_v=features_GSR;
COMPLETE_1_a=features_GSR;
COMPLETE_1_v_ECG1=features_ECG1;
COMPLETE_1_a_ECG1=features_ECG1;
COMPLETE_1_v_ECG2=features_ECG2;
COMPLETE_1_a_ECG2=features_ECG2;
COMPLETE_1_v_EEG1=features_EEG1;
COMPLETE_1_a_EEG1=features_EEG1;
COMPLETE_1_v_EEG15=features_EEG15;
COMPLETE_1_a_EEG15=features_EEG15;
COMPLETE_1_v_EEG2=features_EEG2;
COMPLETE_1_a_EEG2=features_EEG2;
COMPLETE_1_v_EEG25=features_EEG25;
COMPLETE_1_a_EEG25=features_EEG25;
D_full = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]; %dimension of the hypervectors
%D_full = [2000];
randCounter = 50;
acc_matrix = zeros(10);

acc_matrix = [acc_matrix;acc_matrix];
while (randCounter>0)
for j=1:length(D_full)
learningFrac = learningrate(1); 
learningFrac;
D=D_full(j);
D
iMch1_array = zeros(channels_v, D);
projM1_pos = zeros(channels_v, D);
projM1_neg = zeros(channels_v, D);

iMch3_array = zeros(channels_v_ECG1, D);
projM3_pos = zeros(channels_v_ECG1, D);
projM3_neg = zeros(channels_v_ECG1, D);

iMch4_array = zeros(channels_v_ECG2, D);
projM4_pos = zeros(channels_v_ECG2, D);
projM4_neg = zeros(channels_v_ECG2, D);

iMch5_array = zeros(channels_v_EEG1, D);
projM5_pos = zeros(channels_v_EEG1, D);
projM5_neg = zeros(channels_v_EEG1, D);

iMch55_array = zeros(channels_v_EEG15, D);
projM55_pos = zeros(channels_v_EEG15, D);
projM55_neg = zeros(channels_v_EEG15, D);

iMch6_array = zeros(channels_v_EEG2, D);
projM6_pos = zeros(channels_v_EEG2, D);
projM6_neg = zeros(channels_v_EEG2, D);

iMch65_array = zeros(channels_v_EEG25, D);
projM65_pos = zeros(channels_v_EEG25, D);
projM65_neg = zeros(channels_v_EEG25, D);


classes = 2; % level of classes
precision = 20; %no use
ngram = 3; % for temporal encode
maxL = 2; % for IM gen
 
channels_v_EXG=channels_v +channels_v_ECG+channels_v_EEG;
channels_a_EXG=channels_a+channels_a_ECG+channels_a_EEG;


%[chAM1, iMch1] = initItemMemories (D, maxL, channels_v);
%[chAM3, iMch3] = initItemMemories (D, maxL, channels_v_ECG);
%[chAM5, iMch5] = initItemMemories (D, maxL, channels_v_EEG);


combinations_necessary = max([channels_v_EEG1,channels_v_EEG15,channels_v_EEG2,channels_v_EEG25, channels_v_ECG1,channels_v_ECG2, channels_v]);
outputs = 0;
num_vectors = 0;

%counts number of vectors needed for at least 105 combinations
%output is 23 necessary vectors for 110 combinations
while (outputs < combinations_necessary)
    outputs = vector_counter(num_vectors);
    num_vectors = num_vectors + 1;
end

[chAM1, randSetVectors] = initItemMemories (D, maxL, num_vectors);
%generate first iM here for EEG and replicate it into each modality

%110x3 cell array of 1 iM and 2 proj
combinations = final_arrange(values(randSetVectors));

randCounter
c = randCounter;
[s1,s2] = size(combinations);

while (c>0)
    randints = randi([1,s1],1,combinations_necessary);
    c = c-1;
end



%sets first vector of each combination to iM for each channel based on how many features there are
for i = 1:1:channels_v_EEG1
    iMch5_array(i,:) = combinations{randints(i),1};
    projM5_pos(i,:) = combinations{randints(i),2};
    projM5_neg(i,:) = combinations{randints(i),3};
end
randints = randi([1,s1],1,combinations_necessary);
for i = 1:1:channels_v_EEG15
    iMch55_array(i,:) = combinations{randints(i),1};
    projM55_pos(i,:) = combinations{randints(i),2};
    projM55_neg(i,:) = combinations{randints(i),3};
end
randints = randi([1,s1],1,combinations_necessary);

for i = 1:1:channels_v_EEG2
    iMch6_array(i,:) = combinations{randints(i),1};
    projM6_pos(i,:) = combinations{randints(i),2};
    projM6_neg(i,:) = combinations{randints(i),3};
end
randints = randi([1,s1],1,combinations_necessary);

for i = 1:1:channels_v_EEG25
    iMch65_array(i,:) = combinations{randints(i),1};
    projM65_pos(i,:) = combinations{randints(i),2};
    projM65_neg(i,:) = combinations{randints(i),3};
end
randints = randi([1,s1],1,combinations_necessary);

for i=1:1:channels_v_ECG1
    iMch3_array(i,:) = combinations{randints(i),1};
    projM3_pos(i,:) = combinations{randints(i),2};
    projM3_neg(i,:) = combinations{randints(i),3};
end
    randints = randi([1,s1],1,combinations_necessary);
for i=1:1:channels_v_ECG2
    iMch4_array(i,:) = combinations{randints(i),1};
    projM4_pos(i,:) = combinations{randints(i),2};
    projM4_neg(i,:) = combinations{randints(i),3};
end
randints = randi([1,s1],1,combinations_necessary);

for i=1:1:channels_v
    iMch1_array(i,:) = combinations{randints(i),1};
    projM1_pos(i,:) = combinations{randints(i),2};
    projM1_neg(i,:) = combinations{randints(i),3};
end

[chAM7, iMch7] = initItemMemories (D, maxL, channels_v_EXG);
[chAM8, iMch8] = initItemMemories (D, maxL, channels_a_EXG);

%downsample the dataset using the value contained in the variable "downSampRate"
%returns downsampled data which skips every 8 of the original dataset
downSampRate = 8;
LABEL_1_v=f_label_v_binary;
LABEL_1_a=f_label_a_binary;
[TS_COMPLETE_1, L_TS_COMPLETE_1] = downSampling (COMPLETE_1_v, LABEL_1_v, downSampRate);
[TS_COMPLETE_2, L_TS_COMPLETE_2] = downSampling (COMPLETE_1_a, LABEL_1_a, downSampRate);
[TS_COMPLETE_3, L_TS_COMPLETE_3] = downSampling (COMPLETE_1_v_ECG1, LABEL_1_v, downSampRate);
[TS_COMPLETE_32, L_TS_COMPLETE_32] = downSampling (COMPLETE_1_a_ECG1, LABEL_1_v, downSampRate);
[TS_COMPLETE_4, L_TS_COMPLETE_4] = downSampling (COMPLETE_1_v_ECG2, LABEL_1_a, downSampRate);
[TS_COMPLETE_42, L_TS_COMPLETE_42] = downSampling (COMPLETE_1_a_ECG2, LABEL_1_a, downSampRate);
[TS_COMPLETE_5, L_TS_COMPLETE_5] = downSampling (COMPLETE_1_v_EEG1, LABEL_1_v, downSampRate);
[TS_COMPLETE_52, L_TS_COMPLETE_52] = downSampling (COMPLETE_1_a_EEG1, LABEL_1_a, downSampRate);
[TS_COMPLETE_55, L_TS_COMPLETE_55] = downSampling (COMPLETE_1_v_EEG15, LABEL_1_v, downSampRate);
[TS_COMPLETE_552, L_TS_COMPLETE_552] = downSampling (COMPLETE_1_a_EEG15, LABEL_1_a, downSampRate);
[TS_COMPLETE_6, L_TS_COMPLETE_6] = downSampling (COMPLETE_1_v_EEG2, LABEL_1_v, downSampRate);
[TS_COMPLETE_62, L_TS_COMPLETE_62] = downSampling (COMPLETE_1_a_EEG2, LABEL_1_a, downSampRate);
[TS_COMPLETE_65, L_TS_COMPLETE_65] = downSampling (COMPLETE_1_v_EEG25, LABEL_1_v, downSampRate);
[TS_COMPLETE_652, L_TS_COMPLETE_652] = downSampling (COMPLETE_1_a_EEG25, LABEL_1_a, downSampRate);


reduced_TS_COMPLETE_1 = TS_COMPLETE_1;
reduced_TS_COMPLETE_1(reduced_TS_COMPLETE_1 > 0) = 1;
reduced_TS_COMPLETE_1(reduced_TS_COMPLETE_1 < 0) = 2;
reduced_TS_COMPLETE_2 = TS_COMPLETE_2;
reduced_TS_COMPLETE_2(reduced_TS_COMPLETE_2 > 0) = 1;
reduced_TS_COMPLETE_2(reduced_TS_COMPLETE_2 < 0) = 2;
reduced_TS_COMPLETE_3 = TS_COMPLETE_3;
reduced_TS_COMPLETE_3(reduced_TS_COMPLETE_3 > 0) = 1;
reduced_TS_COMPLETE_3(reduced_TS_COMPLETE_3 < 0) = 2;
reduced_TS_COMPLETE_32 = TS_COMPLETE_32;
reduced_TS_COMPLETE_32(reduced_TS_COMPLETE_32 > 0) = 1;
reduced_TS_COMPLETE_32(reduced_TS_COMPLETE_32 < 0) = 2;
reduced_TS_COMPLETE_4 = TS_COMPLETE_4;
reduced_TS_COMPLETE_4(reduced_TS_COMPLETE_4 > 0) = 1;
reduced_TS_COMPLETE_4(reduced_TS_COMPLETE_4 < 0) = 2;
reduced_TS_COMPLETE_42 = TS_COMPLETE_42;
reduced_TS_COMPLETE_42(reduced_TS_COMPLETE_42 > 0) = 1;
reduced_TS_COMPLETE_42(reduced_TS_COMPLETE_42 < 0) = 2;
reduced_TS_COMPLETE_5 = TS_COMPLETE_5;
reduced_TS_COMPLETE_5(reduced_TS_COMPLETE_5 > 0) = 1;
reduced_TS_COMPLETE_5(reduced_TS_COMPLETE_5 < 0) = 2;

reduced_TS_COMPLETE_55 = TS_COMPLETE_55;
reduced_TS_COMPLETE_55(reduced_TS_COMPLETE_55 > 0) = 1;
reduced_TS_COMPLETE_55(reduced_TS_COMPLETE_55 < 0) = 2;

reduced_TS_COMPLETE_52 = TS_COMPLETE_52;
reduced_TS_COMPLETE_52(reduced_TS_COMPLETE_52 > 0) = 1;
reduced_TS_COMPLETE_52(reduced_TS_COMPLETE_52 < 0) = 2;

reduced_TS_COMPLETE_552 = TS_COMPLETE_552;
reduced_TS_COMPLETE_552(reduced_TS_COMPLETE_552 > 0) = 1;
reduced_TS_COMPLETE_552(reduced_TS_COMPLETE_552 < 0) = 2;

reduced_TS_COMPLETE_6 = TS_COMPLETE_6;
reduced_TS_COMPLETE_6(reduced_TS_COMPLETE_6 > 0) = 1;
reduced_TS_COMPLETE_6(reduced_TS_COMPLETE_6 < 0) = 2;

reduced_TS_COMPLETE_62 = TS_COMPLETE_62;
reduced_TS_COMPLETE_62(reduced_TS_COMPLETE_62 > 0) = 1;
reduced_TS_COMPLETE_62(reduced_TS_COMPLETE_62 < 0) = 2;

reduced_TS_COMPLETE_65 = TS_COMPLETE_65;
reduced_TS_COMPLETE_65(reduced_TS_COMPLETE_65 > 0) = 1;
reduced_TS_COMPLETE_65(reduced_TS_COMPLETE_65 < 0) = 2;

reduced_TS_COMPLETE_652 = TS_COMPLETE_652;
reduced_TS_COMPLETE_652(reduced_TS_COMPLETE_652 > 0) = 1;
reduced_TS_COMPLETE_652(reduced_TS_COMPLETE_652 < 0) = 2;

reduced_L_TS_COMPLETE_1 = L_TS_COMPLETE_1;
reduced_L_TS_COMPLETE_1(reduced_L_TS_COMPLETE_1 == 1) = 0;
reduced_L_TS_COMPLETE_1(reduced_L_TS_COMPLETE_1 == 2) = 1;
reduced_L_TS_COMPLETE_2 = L_TS_COMPLETE_2;
reduced_L_TS_COMPLETE_2(reduced_L_TS_COMPLETE_2 == 1) = 0;
reduced_L_TS_COMPLETE_2(reduced_L_TS_COMPLETE_2 == 2) = 1;

%Valence
valence_count_class_change = 0;
for i = 1:1:length(LABEL_1_v)-1
    if LABEL_1_v(i) ~= LABEL_1_v(i+1)
        valence_count_class_change = valence_count_class_change+1;
    end
end
%arousal
arousal_count_class_change = 0;
for i = 1:1:length(LABEL_1_a)-1
    if LABEL_1_a(i) ~= LABEL_1_a(i+1)
        arousal_count_class_change = arousal_count_class_change+1;
    end
end
%generate the training matrices using the learning rate contined in the
%variable "learningFrac"
% 1 = v + GSR
% 2 = a + GSR
% 3 = v + ECG
% 4 = a + ECG
% 5 = v + EEG1
% 52 = a + EEG1
% 6 = v + EEG2
% 62 = a + EEG2

% gen training data finds all the samples corresponding to labels up to 7
% (only see 1 and 2 in the data though). It allocates a certain percentage
% to training data. Then it creates a dataset with labels corresponding to
% the selected data for training. The label dataset is in order from 1-7
% and the data is also stacked one by one so that it is in order from 1-7
[L_SAMPL_DATA_1, SAMPL_DATA_1] = genTrainData (TS_COMPLETE_1, L_TS_COMPLETE_1, learningFrac, 'inorder');
[L_SAMPL_DATA_2, SAMPL_DATA_2] = genTrainData (TS_COMPLETE_2, L_TS_COMPLETE_2, learningFrac, 'inorder');
[L_SAMPL_DATA_3, SAMPL_DATA_3] = genTrainData (TS_COMPLETE_3, L_TS_COMPLETE_3, learningFrac, 'inorder');
[L_SAMPL_DATA_32, SAMPL_DATA_32] = genTrainData (TS_COMPLETE_32, L_TS_COMPLETE_32, learningFrac, 'inorder');
[L_SAMPL_DATA_4, SAMPL_DATA_4] = genTrainData (TS_COMPLETE_4, L_TS_COMPLETE_4, learningFrac, 'inorder');
[L_SAMPL_DATA_42, SAMPL_DATA_42] = genTrainData (TS_COMPLETE_42, L_TS_COMPLETE_42, learningFrac, 'inorder');
[L_SAMPL_DATA_5, SAMPL_DATA_5] = genTrainData (TS_COMPLETE_5, L_TS_COMPLETE_5, learningFrac, 'inorder');
[L_SAMPL_DATA_52, SAMPL_DATA_52] = genTrainData (TS_COMPLETE_52, L_TS_COMPLETE_52, learningFrac, 'inorder');
[L_SAMPL_DATA_55, SAMPL_DATA_55] = genTrainData (TS_COMPLETE_55, L_TS_COMPLETE_55, learningFrac, 'inorder');
[L_SAMPL_DATA_552, SAMPL_DATA_552] = genTrainData (TS_COMPLETE_552, L_TS_COMPLETE_552, learningFrac, 'inorder');
[L_SAMPL_DATA_6, SAMPL_DATA_6] = genTrainData (TS_COMPLETE_6, L_TS_COMPLETE_6, learningFrac, 'inorder');
[L_SAMPL_DATA_62, SAMPL_DATA_62] = genTrainData (TS_COMPLETE_62, L_TS_COMPLETE_62, learningFrac, 'inorder');
[L_SAMPL_DATA_65, SAMPL_DATA_65] = genTrainData (TS_COMPLETE_65, L_TS_COMPLETE_65, learningFrac, 'inorder');
[L_SAMPL_DATA_652, SAMPL_DATA_652] = genTrainData (TS_COMPLETE_652, L_TS_COMPLETE_652, learningFrac, 'inorder');

reduced_SAMPL_DATA_1 = SAMPL_DATA_1;
reduced_SAMPL_DATA_1(reduced_SAMPL_DATA_1 > 0) = 1;
reduced_SAMPL_DATA_1(reduced_SAMPL_DATA_1 < 0) = 2;
reduced_SAMPL_DATA_2 = SAMPL_DATA_2;
reduced_SAMPL_DATA_2(reduced_SAMPL_DATA_2 > 0) = 1;
reduced_SAMPL_DATA_2(reduced_SAMPL_DATA_2 < 0) = 2;
reduced_SAMPL_DATA_3 = SAMPL_DATA_3;
reduced_SAMPL_DATA_3(reduced_SAMPL_DATA_3 > 0) = 1;
reduced_SAMPL_DATA_3(reduced_SAMPL_DATA_3 < 0) = 2;
reduced_SAMPL_DATA_32 = SAMPL_DATA_32;
reduced_SAMPL_DATA_32(reduced_SAMPL_DATA_32 > 0) = 1;
reduced_SAMPL_DATA_32(reduced_SAMPL_DATA_32 < 0) = 2;
reduced_SAMPL_DATA_4 = SAMPL_DATA_4;
reduced_SAMPL_DATA_4(reduced_SAMPL_DATA_4 > 0) = 1;
reduced_SAMPL_DATA_4(reduced_SAMPL_DATA_4 < 0) = 2;
reduced_SAMPL_DATA_42 = SAMPL_DATA_42;
reduced_SAMPL_DATA_42(reduced_SAMPL_DATA_42 > 0) = 1;
reduced_SAMPL_DATA_42(reduced_SAMPL_DATA_42 < 0) = 2;
reduced_SAMPL_DATA_5 = SAMPL_DATA_5;
reduced_SAMPL_DATA_5(reduced_SAMPL_DATA_5 > 0) = 1;
reduced_SAMPL_DATA_5(reduced_SAMPL_DATA_5 < 0) = 2;
reduced_SAMPL_DATA_52 = SAMPL_DATA_52;
reduced_SAMPL_DATA_52(reduced_SAMPL_DATA_52 > 0) = 1;
reduced_SAMPL_DATA_52(reduced_SAMPL_DATA_52 < 0) = 2;
reduced_SAMPL_DATA_6 = SAMPL_DATA_6;
reduced_SAMPL_DATA_6(reduced_SAMPL_DATA_6 > 0) = 1;
reduced_SAMPL_DATA_6(reduced_SAMPL_DATA_6 < 0) = 2;
reduced_SAMPL_DATA_62 = SAMPL_DATA_62;
reduced_SAMPL_DATA_62(reduced_SAMPL_DATA_62 > 0) = 1;
reduced_SAMPL_DATA_62(reduced_SAMPL_DATA_62 < 0) = 2;

reduced_SAMPL_DATA_55 = SAMPL_DATA_55;
reduced_SAMPL_DATA_55(reduced_SAMPL_DATA_55 > 0) = 1;
reduced_SAMPL_DATA_55(reduced_SAMPL_DATA_55 < 0) = 2;
reduced_SAMPL_DATA_552 = SAMPL_DATA_552;
reduced_SAMPL_DATA_552(reduced_SAMPL_DATA_552 > 0) = 1;
reduced_SAMPL_DATA_552(reduced_SAMPL_DATA_552 < 0) = 2;
reduced_SAMPL_DATA_65 = SAMPL_DATA_65;
reduced_SAMPL_DATA_65(reduced_SAMPL_DATA_65 > 0) = 1;
reduced_SAMPL_DATA_65(reduced_SAMPL_DATA_65 < 0) = 2;
reduced_SAMPL_DATA_652 = SAMPL_DATA_652;
reduced_SAMPL_DATA_652(reduced_SAMPL_DATA_652 > 0) = 1;
reduced_SAMPL_DATA_652(reduced_SAMPL_DATA_652 < 0) = 2;


reduced_L_SAMPL_DATA_1 = L_SAMPL_DATA_1;
reduced_L_SAMPL_DATA_1(reduced_L_SAMPL_DATA_1 == 1) = 0;
reduced_L_SAMPL_DATA_1(reduced_L_SAMPL_DATA_1 == 2) = 1;
reduced_L_SAMPL_DATA_2 = L_SAMPL_DATA_2;
reduced_L_SAMPL_DATA_2(reduced_L_SAMPL_DATA_2 == 1) = 0;
reduced_L_SAMPL_DATA_2(reduced_L_SAMPL_DATA_2 == 2) = 1;

%Sparse biopolar mapping
%creates matrix of random hypervectors with element values 1, 0, and -1,
%matrix is has feature (channel) numbers of binary D size hypervectors
%Should be the S vectors
q=0.7;
  

for N = 3:ngram
% creates ngram for data, rotates through and 
N

%NEED TO CONVERT IMS TO MAP CONTAINERS
[x1, iMch1] = initItemMemories (D, maxL, channels_v);
[x3, iMch3] = initItemMemories (D, maxL, channels_v_ECG1);
[x4, iMch4] = initItemMemories (D, maxL, channels_v_ECG2);
[x5, iMch5] = initItemMemories (D, maxL, channels_v_EEG1);
[x6, iMch6] = initItemMemories (D, maxL, channels_v_EEG2);
[x55, iMch55] = initItemMemories (D, maxL, channels_v_EEG15);
[x65, iMch65] = initItemMemories (D, maxL, channels_v_EEG25);

for i=1:1:channels_v
    iMch1(i) = iMch1_array(i,:);
end
for i=1:1:channels_v_ECG1
    iMch3(i) = iMch3_array(i,:);
end
for i=1:1:channels_v_ECG2
    iMch4(i) = iMch4_array(i,:);
end
for i=1:1:channels_v_EEG1
    iMch5(i) = iMch5_array(i,:);
end
for i=1:1:channels_v_EEG15
    iMch55(i) = iMch55_array(i,:);
end
for i=1:1:channels_v_EEG2
    iMch6(i) = iMch6_array(i,:);
end
for i=1:1:channels_v_EEG25
    iMch65(i) = iMch65_array(i,:);
end
% iMch1 = containers.Map(keys(im1),iMch1);
% iMch3 = containers.Map(keys(im3),iMch3);
% iMch5 = containers.Map(keys(im5),iMch5);
% values(iMch1)

% Arousal
%generate ngram bundles for each data stream
fprintf ('HDC for A\n');
if (select == 1)
    [numpat, hdc_model_2] = hdctrainproja (classes, reduced_L_SAMPL_DATA_2,reduced_L_SAMPL_DATA_2,reduced_L_SAMPL_DATA_2, reduced_L_SAMPL_DATA_2, reduced_L_SAMPL_DATA_2, reduced_L_SAMPL_DATA_2,reduced_L_SAMPL_DATA_2,reduced_SAMPL_DATA_2, reduced_SAMPL_DATA_32, reduced_SAMPL_DATA_42, reduced_SAMPL_DATA_52,reduced_SAMPL_DATA_552, reduced_SAMPL_DATA_62,reduced_SAMPL_DATA_652, chAM8, iMch1, iMch3,iMch4, iMch5, iMch55, iMch6, iMch65, D, N, precision, channels_a, channels_a_ECG1,channels_a_ECG2, channels_a_EEG1,channels_a_EEG15, channels_a_EEG2, channels_a_EEG25, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM4_pos, projM4_neg, projM5_pos, projM5_neg, projM55_pos, projM55_neg, projM6_pos, projM6_neg, projM65_pos, projM65_neg); 
else
    [numpat_2, hdc_model_2] = hdctrainproja (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_2, chAM8, iMch1, D, N, precision, channels_a,projM1_pos,projM1_neg, classes); 
    [numpat_32, hdc_model_32] = hdctrainproja (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_32, chAM8, iMch3, D, N, precision, channels_a_ECG1,projM3_pos,projM3_neg, classes); 
    [numpat_42, hdc_model_42] = hdctrainproja (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_42, chAM8, iMch4, D, N, precision, channels_a_ECG2,projM4_pos,projM4_neg, classes); 
    [numpat_52, hdc_model_52] = hdctrainproja (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_52, chAM8, iMch5, D, N, precision, channels_a_EEG1,projM5_pos,projM5_neg, classes); 
    [numpat_552, hdc_model_552] = hdctrainproja (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_552, chAM8, iMch55, D, N, precision, channels_a_EEG1,projM55_pos,projM55_neg, classes); 
    [numpat_62, hdc_model_62] = hdctrainproja (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_62, chAM8, iMch6, D, N, precision, channels_a_EEG2,projM6_pos,projM6_neg, classes); 
    [numpat_652, hdc_model_652] = hdctrainproja (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_652, chAM8, iMch65, D, N, precision, channels_a_EEG1,projM65_pos,projM65_neg, classes); 

end

%bundle all the sensors (this is the fusion point)
if (select ~= 1)
    %class 1
    hdc_model_2(0)=mode([hdc_model_2(0); hdc_model_32(0);hdc_model_42(0); hdc_model_52(0); hdc_model_62(0)]);
    %class 2
    hdc_model_2(1)=mode([hdc_model_2(1);hdc_model_32(1); hdc_model_42(1); hdc_model_52(1); hdc_model_62(1)]);
end

[acc_ex2, acc2, pl2, al2, all_error] = hdcpredictproja  (reduced_L_TS_COMPLETE_2, reduced_TS_COMPLETE_2, reduced_L_TS_COMPLETE_2, reduced_TS_COMPLETE_32, reduced_L_TS_COMPLETE_2, reduced_TS_COMPLETE_42, reduced_L_TS_COMPLETE_2, reduced_TS_COMPLETE_52, reduced_L_TS_COMPLETE_2, reduced_TS_COMPLETE_552, reduced_L_TS_COMPLETE_2, reduced_TS_COMPLETE_62, reduced_L_TS_COMPLETE_2, reduced_TS_COMPLETE_652, hdc_model_2, chAM8, iMch1, iMch3, iMch4, iMch5,iMch55, iMch6,iMch65, D, N, precision, classes, channels_a,channels_a_ECG1, channels_a_ECG2,channels_a_EEG1,channels_a_EEG15, channels_a_EEG2,channels_a_EEG25, projM1_pos, projM1_neg, projM3_pos, projM3_neg,projM4_pos, projM4_neg, projM5_pos, projM5_neg,projM55_pos, projM55_neg, projM6_pos, projM6_neg, projM65_pos, projM65_neg);

accuracy(N,2) = acc2;
acc2
acc_matrix((randCounter*2),(D/1000)) = acc2;
 
%acc_ngram_1(N,j)=acc1;
acc_ngram_A(N,j)=acc2;

% Valence

fprintf ('HDC for V\n');
if (select == 1)
     [numpat, hdc_model_1] = hdctrainproja (classes, reduced_L_SAMPL_DATA_1, reduced_L_SAMPL_DATA_1,  reduced_L_SAMPL_DATA_1,reduced_L_SAMPL_DATA_1,  reduced_L_SAMPL_DATA_1, reduced_L_SAMPL_DATA_1,reduced_SAMPL_DATA_1,reduced_SAMPL_DATA_1, reduced_SAMPL_DATA_3,reduced_SAMPL_DATA_4, reduced_SAMPL_DATA_5,reduced_SAMPL_DATA_55,reduced_SAMPL_DATA_6,reduced_SAMPL_DATA_65, chAM8, iMch1, iMch3,iMch4, iMch5, iMch55,iMch6, iMch65, D, N, precision, channels_v, channels_v_ECG1,channels_v_ECG2, channels_v_EEG1, channels_v_EEG15, channels_v_EEG2,channels_v_EEG25, projM1_pos,projM1_neg, projM3_pos,projM3_neg,projM4_pos,projM4_neg, projM5_pos,projM5_neg,projM55_pos,projM55_neg, projM6_pos, projM6_neg, projM65_pos, projM65_neg); 
else
    [numpat_1, hdc_model_1] = hdctrainproja (reduced_L_SAMPL_DATA_1, reduced_SAMPL_DATA_1, chAM8, iMch1, D, N, precision, channels_v,projM1_pos,projM1_neg, classes); 
    [numpat_3, hdc_model_3] = hdctrainproja (reduced_L_SAMPL_DATA_1, reduced_SAMPL_DATA_3, chAM8, iMch3, D, N, precision, channels_v_ECG1,projM3_pos,projM3_neg, classes); 
    [numpat_4, hdc_model_4] = hdctrainproja (reduced_L_SAMPL_DATA_1, reduced_SAMPL_DATA_4, chAM8, iMch4, D, N, precision, channels_v_ECG2,projM4_pos,projM4_neg, classes); 
    [numpat_5, hdc_model_5] = hdctrainproja (reduced_L_SAMPL_DATA_1, reduced_SAMPL_DATA_5, chAM8, iMch5, D, N, precision, channels_v_EEG1,projM5_pos,projM5_neg, classes); 
    [numpat_55, hdc_model_55] = hdctrainproja (reduced_L_SAMPL_DATA_1, reduced_SAMPL_DATA_55, chAM8, iMch55, D, N, precision, channels_v_EEG15,projM55_pos,projM55_neg, classes); 
    [numpat_6, hdc_model_6] = hdctrainproja (reduced_L_SAMPL_DATA_1, reduced_SAMPL_DATA_6, chAM8, iMch6, D, N, precision, channels_v_EEG2,projM6_pos,projM6_neg, classes); 
    [numpat_65, hdc_model_65] = hdctrainproja (reduced_L_SAMPL_DATA_1, reduced_SAMPL_DATA_65, chAM8, iMch65, D, N, precision, channels_v_EEG25,projM65_pos,projM65_neg, classes); 

end

if (select ~= 1)
    %class 1
    hdc_model_1(0)=mode([hdc_model_1(0); hdc_model_3(0);hdc_model_4(0); hdc_model_5(0); hdc_model_6(0)]);
    %class 2
    hdc_model_1(1)=mode([hdc_model_1(1); hdc_model_3(1);hdc_model_4(1); hdc_model_5(1); hdc_model_6(1)]);
end


[acc_ex1, acc1, pl1, al1, all_error] = hdcpredictproja  (reduced_L_TS_COMPLETE_1, reduced_TS_COMPLETE_1, reduced_L_TS_COMPLETE_1, reduced_TS_COMPLETE_3,reduced_L_TS_COMPLETE_1, reduced_TS_COMPLETE_4, reduced_L_TS_COMPLETE_1, reduced_TS_COMPLETE_5,reduced_L_TS_COMPLETE_1, reduced_TS_COMPLETE_55, reduced_L_TS_COMPLETE_1, reduced_TS_COMPLETE_6, reduced_L_TS_COMPLETE_1, reduced_TS_COMPLETE_65, hdc_model_1, chAM8, iMch1, iMch3,iMch4, iMch5, iMch55, iMch6, iMch65, D, N, precision, classes, channels_v,channels_v_ECG1,channels_v_ECG2,channels_v_EEG1,channels_v_EEG15, channels_v_EEG2, channels_v_EEG25, projM1_pos,projM1_neg,projM3_pos,projM3_neg,projM4_pos, projM4_neg,projM5_pos,projM5_neg, projM55_pos, projM55_neg, projM6_pos, projM6_neg, projM65_pos, projM65_neg);
%for verification
%[acc_ex1, acc1, pl1, al1, all_error] = hdcpredictproj  (L_SAMPL_DATA_1, SAMPL_DATA_1, L_SAMPL_DATA_3, SAMPL_DATA_3, L_SAMPL_DATA_5, SAMPL_DATA_5,hdc_model, chAM8, iMch1, iMch3, iMch5, D, N, precision, classes, channels_v,channels_v_ECG,channels_v_EEG,projM1,projM3,projM5);

accuracy(N,2) = acc1;
acc1
acc_matrix((randCounter*2-1),(D/1000)) = acc1;

%acc_ngram_1(N,j)=acc1;
acc_ngram_V(N,j)=acc1;
end

for i = 1:1:length(1:ngram)
    if acc_ngram_A(i,j) > 0.684
        met_A_accuracy(j) = 1;
    else
        
    end
    if acc_ngram_V(i,j) > 0.801
        met_V_accuracy(j) = 1;
    end
end

end

iMfull = [];
for i = 1:1:iMch5.Count
    iMfull = [iMfull iMch5(i)]; %#ok<AGROW>
end

projM_pos_full = [];
%projM_pos_temp = [projM1_pos; projM3_pos; projM5_pos];
x = size(projM5_pos);
dim = x(1);
for i = 1:1:dim
    projM_pos_full = [projM_pos_full projM5_pos(i,:)]; %#ok<AGROW>
end

projM_neg_full = [];
%projM_neg_temp = [projM1_neg; projM3_neg; projM5_neg];
x = size(projM5_neg);
dim = x(1);
for i = 1:1:dim
    projM_neg_full = [projM_neg_full projM5_neg(i,:)]; %#ok<AGROW>
end

randCounter=randCounter-1;
%given the first vector, creates all possible vector combinations
end
acc_matrix
function vec_array = arrange_vectors(m)  
    vec_array = [];
    m_copy = m;
    if (mod(length(m_copy), 2)== 1)
        m_copy(end) = []; 
    end
    while (length(m_copy) > 2)
        last = m_copy(end);
        m_copy(end) = [];   
        last2 = m_copy(end);
        m_copy(end) = [];
        arr = [last, last2];
        vec_array = [vec_array; [m_copy(1,1), arr]];
    end
    
end

%uses arrange_vectors on a list of vectors
function complete_array = final_arrange(m)
    complete_array = [];
    while (length(m)>2)
        complete_array = [complete_array; arrange_vectors(m)];
        m(1) = [];
    end
end



%given a number of vectors, counts all combinations
function num_vectors = vector_counter(x)
num_vectors = 0;    
subtracter = 1;
    while (subtracter < (x-1))
        num_vectors = num_vectors + floor((x-subtracter)/2);
        subtracter = subtracter + 1;
    
    end
end
