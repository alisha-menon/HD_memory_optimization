clear;

%====Features and Label===
load('DEAP_data.mat')
features=inputs;
f_label_a_binary=features(:,239);
f_label_v_binary=features(:,240);
f_label_d_binary=features(:,241);
f_label_l_binary=features(:,242);

m = 4.5;

% f_label_a_binary(f_label_a_binary < m/2) = 1;
% f_label_a_binary(m/2 <= f_label_a_binary & f_label_a_binary < m) = 2;
% f_label_a_binary(m <= f_label_a_binary & f_label_a_binary < 3*m/2) = 3;
% f_label_a_binary(f_label_a_binary >= 3*m/2) = 4;
% f_label_v_binary(f_label_v_binary < m/2) = 1;
% f_label_v_binary(m/2 < f_label_v_binary & f_label_v_binary < m) = 2;
% f_label_v_binary(m < f_label_v_binary & f_label_v_binary < 3*m/2) = 3;
% f_label_v_binary(f_label_v_binary >= 3*m/2) = 4;
% f_label_d_binary(f_label_d_binary < m/2) = 1;
% f_label_d_binary(m/2 < f_label_d_binary & f_label_d_binary < m) = 2;
% f_label_d_binary(m < f_label_d_binary & f_label_d_binary < 3*m/2) = 3;
% f_label_d_binary(f_label_d_binary >= 3*m/2) = 4;
% f_label_l_binary(f_label_l_binary < m/2) = 1;
% f_label_l_binary(m/2 < f_label_l_binary & f_label_l_binary < m) = 2;
% f_label_l_binary(m < f_label_l_binary & f_label_l_binary < 3*m/2) = 3;
% f_label_l_binary(f_label_l_binary >= 3*m/2) = 4;

f_label_a_binary(f_label_a_binary < m) = 1;
f_label_a_binary(f_label_a_binary >= m) = 2;
f_label_v_binary(f_label_v_binary < m) = 1;
f_label_v_binary(f_label_v_binary >= m) = 2;
f_label_d_binary(f_label_d_binary < m) = 1;
f_label_d_binary(f_label_d_binary >= m) = 2;
f_label_l_binary(f_label_l_binary < m) = 1;
f_label_l_binary(f_label_l_binary >= m) = 2;

met_A_accuracy = zeros(1,12);
met_V_accuracy = zeros(1,12);
met_D_accuracy = zeros(1,12);
met_L_accuracy = zeros(1,12);

for k=1:238
 features(:,k)=features(:,k)-min(features(:,k));
end

for k=1:238
 features(:,k)=features(:,k)/max(features(:,k));
end

for i=1:238
 features(:,i)=2*features(:,i)-1;
end

for i=1:238
 features(:,i)=2*features(:,i)+0.35;
end

features_EMG=features(:,1:10);
features_EEG=features(:,11:202);
features_GSR=features(:,203:209);
features_BVP=features(:,210:226);
features_RES=features(:,227:238);


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


channels_v=length(features_EMG(1,:));
channels_v_EEG=length(features_EEG(1,:));
channels_v_GSR=length(features_GSR(1,:));
channels_v_BVP=length(features_BVP(1,:));
channels_v_RES=length(features_RES(1,:));

channels_a=channels_v;
channels_a_EEG=channels_v_EEG;
channels_a_GSR=channels_v_GSR;
channels_a_BVP=channels_v_BVP;
channels_a_RES=channels_v_RES;

channels_d=channels_v;
channels_d_EEG=channels_v_EEG;
channels_d_GSR=channels_v_GSR;
channels_d_BVP=channels_v_BVP;
channels_d_RES=channels_v_RES;

channels_l=channels_v;
channels_l_EEG=channels_v_EEG;
channels_l_GSR=channels_v_GSR;
channels_l_BVP=channels_v_BVP;
channels_l_RES=channels_v_RES;

COMPLETE_1_v=features_EMG;
COMPLETE_1_a=features_EMG;
COMPLETE_1_d=features_EMG;
COMPLETE_1_l=features_EMG;

COMPLETE_1_v_EEG=features_EEG;
COMPLETE_1_a_EEG=features_EEG;
COMPLETE_1_d_EEG=features_EEG;
COMPLETE_1_l_EEG=features_EEG;

COMPLETE_1_v_GSR=features_GSR;
COMPLETE_1_a_GSR=features_GSR;
COMPLETE_1_d_GSR=features_GSR;
COMPLETE_1_l_GSR=features_GSR;

COMPLETE_1_v_BVP=features_BVP;
COMPLETE_1_a_BVP=features_BVP;
COMPLETE_1_d_BVP=features_BVP;
COMPLETE_1_l_BVP=features_BVP;

COMPLETE_1_v_RES=features_RES;
COMPLETE_1_a_RES=features_RES;
COMPLETE_1_d_RES=features_RES;
COMPLETE_1_l_RES=features_RES;

% D_full = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]; %dimension of the hypervectors
D_full = [10000];
randCounter= 8;
acc_matrix = zeros(10);
acc_matrix = [acc_matrix;acc_matrix;acc_matrix;acc_matrix];
while (randCounter>0)
for j=1:length(D_full)
learningFrac = learningrate(1); 
learningFrac;
D=D_full(j);
D
iMch1_array = zeros(channels_v, D);
projM1_pos = zeros(channels_v, D);
projM1_neg = zeros(channels_v, D);

iMch3_array = zeros(channels_v_EEG, D);
projM3_pos = zeros(channels_v_EEG, D);
projM3_neg = zeros(channels_v_EEG, D);

iMch5_array = zeros(channels_v_GSR, D);
projM5_pos = zeros(channels_v_GSR, D);
projM5_neg = zeros(channels_v_GSR, D);

iMch7_array = zeros(channels_v_BVP, D);
projM7_pos = zeros(channels_v_BVP, D);
projM7_neg = zeros(channels_v_BVP, D);

iMch9_array = zeros(channels_v_RES, D);
projM9_pos = zeros(channels_v_RES, D);
projM9_neg = zeros(channels_v_RES, D);

classes = 2; % level of classes
precision = 20; %no use
ngram = 3; % for temporal encode
maxL = 2; % for IM gen
 
channels_v_EXG=channels_v+channels_v_EEG+channels_v_GSR+channels_v_BVP+channels_v_RES;
channels_a_EXG=channels_a+channels_a_EEG+channels_a_GSR+channels_a_BVP+channels_a_RES;
channels_d_EXG=channels_d+channels_d_EEG+channels_d_GSR+channels_d_BVP+channels_d_RES;
channels_l_EXG=channels_l+channels_l_EEG+channels_l_GSR+channels_l_BVP+channels_l_RES;


%[chAM1, iMch1] = initItemMemories (D, maxL, channels_v);
%[chAM3, iMch3] = initItemMemories (D, maxL, channels_v_ECG);
%[chAM5, iMch5] = initItemMemories (D, maxL, channels_v_EEG);


combinations_necessary = max([channels_v_EEG, channels_v_GSR, channels_v, channels_v_BVP, channels_v_RES]);
outputs = 0;
num_vectors = 0;

%counts number of vectors needed for at least 105 combinations
%output is 23 necessary vectors for 110 combinations
while (outputs < combinations_necessary)
    outputs = vector_counter(num_vectors);
    num_vectors = num_vectors + 1;
end


%HERE IS WHAT WE NEED TO CHANGE
% [chAM1, randSetVectors] = initItemMemories (D, maxL, num_vectors);

% RULE 90
% total_vectors = 23;
% seed_vector = genRandomHV(D);
% randSetVectors  = containers.Map ('KeyType','double','ValueType','any');
% for i = 1:total_vectors
%     seed_vector = xor(circshift(seed_vector,1),circshift(seed_vector,-1));
%     randSetVectors(i) = seed_vector;
% end

% Cellular Automata
%30, 45
rule = 30;
neighborhood_size = 3;
seed_vector = genRandomHV(D);
total_vectors = num_vectors;

offset = floor(neighborhood_size/2);
ruleset = fliplr(dec2bin(rule,2^neighborhood_size));
randSetVectors  = containers.Map ('KeyType','double','ValueType','any');
for i = 1:total_vectors
    new_vector = zeros(1,D);
    for d = 1:D
        if d-offset<1
            neighborhood = [seed_vector(D-(offset-d):D) seed_vector(1:d+offset)];
        elseif d+offset>D
            neighborhood = [seed_vector(d-offset:D) seed_vector(1:offset-(D-d))];
        else
            neighborhood = seed_vector(d-offset:d+offset);
        end
        index = bin2dec(num2str(neighborhood));
        new_vector(d) = ruleset(index+1)-48;
    end
    seed_vector = new_vector;
    randSetVectors(i) = seed_vector;
end



%generate first iM here for EEG and replicate it into each modality

%110x3 cell array of 1 iM and 2 proj
combinations = final_arrange(values(randSetVectors));

randCounter
[s1,s2] = size(combinations);

c = randCounter;
while (c>0)
    randints = randi([1,s1],1,combinations_necessary);
    c = c-1;
end



%sets first vector of each combination to iM for each channel based on how many features there are
for i = 1:1:channels_v_RES
    iMch9_array(i,:) = combinations{randints(i),1};
    projM9_pos(i,:) = combinations{randints(i),2};
    projM9_neg(i,:) = combinations{randints(i),3};
end
    randints = randi([1,s1],1,combinations_necessary);

for i = 1:1:channels_v_BVP
    iMch7_array(i,:) = combinations{randints(i),1};
    projM7_pos(i,:) = combinations{randints(i),2};
    projM7_neg(i,:) = combinations{randints(i),3};
end
    randints = randi([1,s1],1,combinations_necessary);

for i = 1:1:channels_v_GSR
    iMch5_array(i,:) = combinations{randints(i),1};
    projM5_pos(i,:) = combinations{randints(i),2};
    projM5_neg(i,:) = combinations{randints(i),3};
end
    randints = randi([1,s1],1,combinations_necessary);

for i=1:1:channels_v_EEG
    iMch3_array(i,:) = combinations{randints(i),1};
    projM3_pos(i,:) = combinations{randints(i),2};
    projM3_neg(i,:) = combinations{randints(i),3};
end
    randints = randi([1,s1],1,combinations_necessary);

for i=1:1:channels_v
    iMch1_array(i,:) = combinations{randints(i),1};
    projM1_pos(i,:) = combinations{randints(i),2};
    projM1_neg(i,:) = combinations{randints(i),3};
end

[chAM11, iMch11] = initItemMemories (D, maxL, channels_v_EXG);
[chAM12, iMch12] = initItemMemories (D, maxL, channels_a_EXG);
[chAM13, iMch13] = initItemMemories (D, maxL, channels_d_EXG);
[chAM14, iMch14] = initItemMemories (D, maxL, channels_l_EXG);

%downsample the dataset using the value contained in the variable "downSampRate"
%returns downsampled data which skips every 8 of the original dataset
downSampRate = 8;
LABEL_1_v=f_label_v_binary;
LABEL_1_a=f_label_a_binary;
LABEL_1_d=f_label_d_binary;
LABEL_1_l=f_label_l_binary;

[TS_COMPLETE_01, L_TS_COMPLETE_01] = downSampling (COMPLETE_1_v, LABEL_1_v, downSampRate);
[TS_COMPLETE_02, L_TS_COMPLETE_02] = downSampling (COMPLETE_1_a, LABEL_1_a, downSampRate);
[TS_COMPLETE_03, L_TS_COMPLETE_03] = downSampling (COMPLETE_1_d, LABEL_1_d, downSampRate);
[TS_COMPLETE_04, L_TS_COMPLETE_04] = downSampling (COMPLETE_1_l, LABEL_1_l, downSampRate);

[TS_COMPLETE_11, L_TS_COMPLETE_11] = downSampling (COMPLETE_1_v_EEG, LABEL_1_v, downSampRate);
[TS_COMPLETE_12, L_TS_COMPLETE_12] = downSampling (COMPLETE_1_a_EEG, LABEL_1_a, downSampRate);
[TS_COMPLETE_13, L_TS_COMPLETE_13] = downSampling (COMPLETE_1_d_EEG, LABEL_1_d, downSampRate);
[TS_COMPLETE_14, L_TS_COMPLETE_14] = downSampling (COMPLETE_1_l_EEG, LABEL_1_l, downSampRate);

[TS_COMPLETE_21, L_TS_COMPLETE_21] = downSampling (COMPLETE_1_v_GSR, LABEL_1_v, downSampRate);
[TS_COMPLETE_22, L_TS_COMPLETE_22] = downSampling (COMPLETE_1_a_GSR, LABEL_1_a, downSampRate);
[TS_COMPLETE_23, L_TS_COMPLETE_23] = downSampling (COMPLETE_1_d_GSR, LABEL_1_d, downSampRate);
[TS_COMPLETE_24, L_TS_COMPLETE_24] = downSampling (COMPLETE_1_l_GSR, LABEL_1_l, downSampRate);

[TS_COMPLETE_31, L_TS_COMPLETE_31] = downSampling (COMPLETE_1_v_BVP, LABEL_1_v, downSampRate);
[TS_COMPLETE_32, L_TS_COMPLETE_32] = downSampling (COMPLETE_1_a_BVP, LABEL_1_a, downSampRate);
[TS_COMPLETE_33, L_TS_COMPLETE_33] = downSampling (COMPLETE_1_d_BVP, LABEL_1_d, downSampRate);
[TS_COMPLETE_34, L_TS_COMPLETE_34] = downSampling (COMPLETE_1_l_BVP, LABEL_1_l, downSampRate);

[TS_COMPLETE_41, L_TS_COMPLETE_41] = downSampling (COMPLETE_1_v_RES, LABEL_1_v, downSampRate);
[TS_COMPLETE_42, L_TS_COMPLETE_42] = downSampling (COMPLETE_1_a_RES, LABEL_1_a, downSampRate);
[TS_COMPLETE_43, L_TS_COMPLETE_43] = downSampling (COMPLETE_1_d_RES, LABEL_1_d, downSampRate);
[TS_COMPLETE_44, L_TS_COMPLETE_44] = downSampling (COMPLETE_1_l_RES, LABEL_1_l, downSampRate);


reduced_TS_COMPLETE_01 = TS_COMPLETE_01;
reduced_TS_COMPLETE_01(reduced_TS_COMPLETE_01 > 0) = 1;
reduced_TS_COMPLETE_01(reduced_TS_COMPLETE_01 < 0) = 2;
reduced_TS_COMPLETE_02 = TS_COMPLETE_02;
reduced_TS_COMPLETE_02(reduced_TS_COMPLETE_02 > 0) = 1;
reduced_TS_COMPLETE_02(reduced_TS_COMPLETE_02 < 0) = 2;
reduced_TS_COMPLETE_03 = TS_COMPLETE_03;
reduced_TS_COMPLETE_03(reduced_TS_COMPLETE_03 > 0) = 1;
reduced_TS_COMPLETE_03(reduced_TS_COMPLETE_03 < 0) = 2;
reduced_TS_COMPLETE_04 = TS_COMPLETE_04;
reduced_TS_COMPLETE_04(reduced_TS_COMPLETE_04 > 0) = 1;
reduced_TS_COMPLETE_04(reduced_TS_COMPLETE_04 < 0) = 2;
reduced_TS_COMPLETE_11 = TS_COMPLETE_11;
reduced_TS_COMPLETE_11(reduced_TS_COMPLETE_11 > 0) = 1;
reduced_TS_COMPLETE_11(reduced_TS_COMPLETE_11 < 0) = 2;
reduced_TS_COMPLETE_12 = TS_COMPLETE_12;
reduced_TS_COMPLETE_12(reduced_TS_COMPLETE_12 > 0) = 1;
reduced_TS_COMPLETE_12(reduced_TS_COMPLETE_12 < 0) = 2;
reduced_TS_COMPLETE_13 = TS_COMPLETE_13;
reduced_TS_COMPLETE_13(reduced_TS_COMPLETE_13 > 0) = 1;
reduced_TS_COMPLETE_13(reduced_TS_COMPLETE_13 < 0) = 2;
reduced_TS_COMPLETE_14 = TS_COMPLETE_14;
reduced_TS_COMPLETE_14(reduced_TS_COMPLETE_14 > 0) = 1;
reduced_TS_COMPLETE_14(reduced_TS_COMPLETE_14 < 0) = 2;
reduced_TS_COMPLETE_21 = TS_COMPLETE_21;
reduced_TS_COMPLETE_21(reduced_TS_COMPLETE_21 > 0) = 1;
reduced_TS_COMPLETE_21(reduced_TS_COMPLETE_21 < 0) = 2;
reduced_TS_COMPLETE_22 = TS_COMPLETE_22;
reduced_TS_COMPLETE_22(reduced_TS_COMPLETE_22 > 0) = 1;
reduced_TS_COMPLETE_22(reduced_TS_COMPLETE_22 < 0) = 2;
reduced_TS_COMPLETE_23 = TS_COMPLETE_23;
reduced_TS_COMPLETE_23(reduced_TS_COMPLETE_23 > 0) = 1;
reduced_TS_COMPLETE_23(reduced_TS_COMPLETE_23 < 0) = 2;
reduced_TS_COMPLETE_24 = TS_COMPLETE_24;
reduced_TS_COMPLETE_24(reduced_TS_COMPLETE_24 > 0) = 1;
reduced_TS_COMPLETE_24(reduced_TS_COMPLETE_24 < 0) = 2;
reduced_TS_COMPLETE_31 = TS_COMPLETE_31;
reduced_TS_COMPLETE_31(reduced_TS_COMPLETE_31 > 0) = 1;
reduced_TS_COMPLETE_31(reduced_TS_COMPLETE_31 < 0) = 2;
reduced_TS_COMPLETE_32 = TS_COMPLETE_32;
reduced_TS_COMPLETE_32(reduced_TS_COMPLETE_32 > 0) = 1;
reduced_TS_COMPLETE_32(reduced_TS_COMPLETE_32 < 0) = 2;
reduced_TS_COMPLETE_33 = TS_COMPLETE_33;
reduced_TS_COMPLETE_33(reduced_TS_COMPLETE_33 > 0) = 1;
reduced_TS_COMPLETE_33(reduced_TS_COMPLETE_33 < 0) = 2;
reduced_TS_COMPLETE_34 = TS_COMPLETE_34;
reduced_TS_COMPLETE_34(reduced_TS_COMPLETE_34 > 0) = 1;
reduced_TS_COMPLETE_34(reduced_TS_COMPLETE_34 < 0) = 2;
reduced_TS_COMPLETE_41 = TS_COMPLETE_41;
reduced_TS_COMPLETE_41(reduced_TS_COMPLETE_41 > 0) = 1;
reduced_TS_COMPLETE_41(reduced_TS_COMPLETE_41 < 0) = 2;
reduced_TS_COMPLETE_42 = TS_COMPLETE_42;
reduced_TS_COMPLETE_42(reduced_TS_COMPLETE_42 > 0) = 1;
reduced_TS_COMPLETE_42(reduced_TS_COMPLETE_42 < 0) = 2;
reduced_TS_COMPLETE_43 = TS_COMPLETE_43;
reduced_TS_COMPLETE_43(reduced_TS_COMPLETE_43 > 0) = 1;
reduced_TS_COMPLETE_43(reduced_TS_COMPLETE_43 < 0) = 2;
reduced_TS_COMPLETE_44 = TS_COMPLETE_44;
reduced_TS_COMPLETE_44(reduced_TS_COMPLETE_44 > 0) = 1;
reduced_TS_COMPLETE_44(reduced_TS_COMPLETE_44 < 0) = 2;


reduced_L_TS_COMPLETE_1 = L_TS_COMPLETE_01;
reduced_L_TS_COMPLETE_1(reduced_L_TS_COMPLETE_1 == 1) = 0;
reduced_L_TS_COMPLETE_1(reduced_L_TS_COMPLETE_1 == 2) = 1;
reduced_L_TS_COMPLETE_2 = L_TS_COMPLETE_02;
reduced_L_TS_COMPLETE_2(reduced_L_TS_COMPLETE_2 == 1) = 0;
reduced_L_TS_COMPLETE_2(reduced_L_TS_COMPLETE_2 == 2) = 1;
reduced_L_TS_COMPLETE_3 = L_TS_COMPLETE_03;
reduced_L_TS_COMPLETE_3(reduced_L_TS_COMPLETE_3 == 1) = 0;
reduced_L_TS_COMPLETE_3(reduced_L_TS_COMPLETE_3 == 2) = 1;
reduced_L_TS_COMPLETE_4 = L_TS_COMPLETE_04;
reduced_L_TS_COMPLETE_4(reduced_L_TS_COMPLETE_4 == 1) = 0;
reduced_L_TS_COMPLETE_4(reduced_L_TS_COMPLETE_4 == 2) = 1;

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

dominance_count_class_change = 0;
for i = 1:1:length(LABEL_1_d)-1
    if LABEL_1_d(i) ~= LABEL_1_d(i+1)
        dominance_count_class_change = dominance_count_class_change+1;
    end
end

liking_count_class_change = 0;
for i = 1:1:length(LABEL_1_l)-1
    if LABEL_1_l(i) ~= LABEL_1_l(i+1)
        liking_count_class_change = liking_count_class_change+1;
    end
end
%generate the training matrices using the learning rate contined in the
%variable "learningFrac"
% 1 = v + GSR
% 2 = a + GSR
% 3 = v + ECG
% 4 = a + ECG
% 5 = v + EEG
% 6 = a + EEG
% gen training data finds all the samples corresponding to labels up to 7
% (only see 1 and 2 in the data though). It allocates a certain percentage
% to training data. Then it creates a dataset with labels corresponding to
% the selected data for training. The label dataset is in order from 1-7
% and the data is also stacked one by one so that it is in order from 1-7
[L_SAMPL_DATA_01, SAMPL_DATA_01] = genTrainData (TS_COMPLETE_01, L_TS_COMPLETE_01, learningFrac, 'inorder');
[L_SAMPL_DATA_02, SAMPL_DATA_02] = genTrainData (TS_COMPLETE_02, L_TS_COMPLETE_02, learningFrac, 'inorder');
[L_SAMPL_DATA_03, SAMPL_DATA_03] = genTrainData (TS_COMPLETE_03, L_TS_COMPLETE_03, learningFrac, 'inorder');
[L_SAMPL_DATA_04, SAMPL_DATA_04] = genTrainData (TS_COMPLETE_04, L_TS_COMPLETE_04, learningFrac, 'inorder');
[L_SAMPL_DATA_11, SAMPL_DATA_11] = genTrainData (TS_COMPLETE_11, L_TS_COMPLETE_11, learningFrac, 'inorder');
[L_SAMPL_DATA_12, SAMPL_DATA_12] = genTrainData (TS_COMPLETE_12, L_TS_COMPLETE_12, learningFrac, 'inorder');
[L_SAMPL_DATA_13, SAMPL_DATA_13] = genTrainData (TS_COMPLETE_13, L_TS_COMPLETE_13, learningFrac, 'inorder');
[L_SAMPL_DATA_14, SAMPL_DATA_14] = genTrainData (TS_COMPLETE_14, L_TS_COMPLETE_14, learningFrac, 'inorder');
[L_SAMPL_DATA_21, SAMPL_DATA_21] = genTrainData (TS_COMPLETE_21, L_TS_COMPLETE_21, learningFrac, 'inorder');
[L_SAMPL_DATA_22, SAMPL_DATA_22] = genTrainData (TS_COMPLETE_22, L_TS_COMPLETE_22, learningFrac, 'inorder');
[L_SAMPL_DATA_23, SAMPL_DATA_23] = genTrainData (TS_COMPLETE_23, L_TS_COMPLETE_23, learningFrac, 'inorder');
[L_SAMPL_DATA_24, SAMPL_DATA_24] = genTrainData (TS_COMPLETE_24, L_TS_COMPLETE_24, learningFrac, 'inorder');
[L_SAMPL_DATA_31, SAMPL_DATA_31] = genTrainData (TS_COMPLETE_31, L_TS_COMPLETE_31, learningFrac, 'inorder');
[L_SAMPL_DATA_32, SAMPL_DATA_32] = genTrainData (TS_COMPLETE_32, L_TS_COMPLETE_32, learningFrac, 'inorder');
[L_SAMPL_DATA_33, SAMPL_DATA_33] = genTrainData (TS_COMPLETE_33, L_TS_COMPLETE_33, learningFrac, 'inorder');
[L_SAMPL_DATA_34, SAMPL_DATA_34] = genTrainData (TS_COMPLETE_34, L_TS_COMPLETE_34, learningFrac, 'inorder');
[L_SAMPL_DATA_41, SAMPL_DATA_41] = genTrainData (TS_COMPLETE_41, L_TS_COMPLETE_41, learningFrac, 'inorder');
[L_SAMPL_DATA_42, SAMPL_DATA_42] = genTrainData (TS_COMPLETE_42, L_TS_COMPLETE_42, learningFrac, 'inorder');
[L_SAMPL_DATA_43, SAMPL_DATA_43] = genTrainData (TS_COMPLETE_43, L_TS_COMPLETE_43, learningFrac, 'inorder');
[L_SAMPL_DATA_44, SAMPL_DATA_44] = genTrainData (TS_COMPLETE_44, L_TS_COMPLETE_44, learningFrac, 'inorder');

reduced_SAMPL_DATA_01 = SAMPL_DATA_01;
reduced_SAMPL_DATA_01(reduced_SAMPL_DATA_01 > 0) = 1;
reduced_SAMPL_DATA_01(reduced_SAMPL_DATA_01 < 0) = 2;
reduced_SAMPL_DATA_02 = SAMPL_DATA_02;
reduced_SAMPL_DATA_02(reduced_SAMPL_DATA_02 > 0) = 1;
reduced_SAMPL_DATA_02(reduced_SAMPL_DATA_02 < 0) = 2;
reduced_SAMPL_DATA_03 = SAMPL_DATA_03;
reduced_SAMPL_DATA_03(reduced_SAMPL_DATA_03 > 0) = 1;
reduced_SAMPL_DATA_03(reduced_SAMPL_DATA_03 < 0) = 2;
reduced_SAMPL_DATA_04 = SAMPL_DATA_04;
reduced_SAMPL_DATA_04(reduced_SAMPL_DATA_04 > 0) = 1;
reduced_SAMPL_DATA_04(reduced_SAMPL_DATA_04 < 0) = 2;
reduced_SAMPL_DATA_11 = SAMPL_DATA_11;
reduced_SAMPL_DATA_11(reduced_SAMPL_DATA_11 > 0) = 1;
reduced_SAMPL_DATA_11(reduced_SAMPL_DATA_11 < 0) = 2;
reduced_SAMPL_DATA_12 = SAMPL_DATA_12;
reduced_SAMPL_DATA_12(reduced_SAMPL_DATA_12 > 0) = 1;
reduced_SAMPL_DATA_12(reduced_SAMPL_DATA_12 < 0) = 2;
reduced_SAMPL_DATA_13 = SAMPL_DATA_13;
reduced_SAMPL_DATA_13(reduced_SAMPL_DATA_13 > 0) = 1;
reduced_SAMPL_DATA_13(reduced_SAMPL_DATA_13 < 0) = 2;
reduced_SAMPL_DATA_14 = SAMPL_DATA_14;
reduced_SAMPL_DATA_14(reduced_SAMPL_DATA_14 > 0) = 1;
reduced_SAMPL_DATA_14(reduced_SAMPL_DATA_14 < 0) = 2;
reduced_SAMPL_DATA_21 = SAMPL_DATA_21;
reduced_SAMPL_DATA_21(reduced_SAMPL_DATA_21 > 0) = 1;
reduced_SAMPL_DATA_21(reduced_SAMPL_DATA_21 < 0) = 2;
reduced_SAMPL_DATA_22 = SAMPL_DATA_22;
reduced_SAMPL_DATA_22(reduced_SAMPL_DATA_22 > 0) = 1;
reduced_SAMPL_DATA_22(reduced_SAMPL_DATA_22 < 0) = 2;
reduced_SAMPL_DATA_23 = SAMPL_DATA_23;
reduced_SAMPL_DATA_23(reduced_SAMPL_DATA_23 > 0) = 1;
reduced_SAMPL_DATA_23(reduced_SAMPL_DATA_23 < 0) = 2;
reduced_SAMPL_DATA_24 = SAMPL_DATA_24;
reduced_SAMPL_DATA_24(reduced_SAMPL_DATA_24 > 0) = 1;
reduced_SAMPL_DATA_24(reduced_SAMPL_DATA_24 < 0) = 2;
reduced_SAMPL_DATA_31 = SAMPL_DATA_31;
reduced_SAMPL_DATA_31(reduced_SAMPL_DATA_31 > 0) = 1;
reduced_SAMPL_DATA_31(reduced_SAMPL_DATA_31 < 0) = 2;
reduced_SAMPL_DATA_32 = SAMPL_DATA_32;
reduced_SAMPL_DATA_32(reduced_SAMPL_DATA_32 > 0) = 1;
reduced_SAMPL_DATA_32(reduced_SAMPL_DATA_32 < 0) = 2;
reduced_SAMPL_DATA_33 = SAMPL_DATA_33;
reduced_SAMPL_DATA_33(reduced_SAMPL_DATA_33 > 0) = 1;
reduced_SAMPL_DATA_33(reduced_SAMPL_DATA_33 < 0) = 2;
reduced_SAMPL_DATA_34 = SAMPL_DATA_34;
reduced_SAMPL_DATA_34(reduced_SAMPL_DATA_34 > 0) = 1;
reduced_SAMPL_DATA_34(reduced_SAMPL_DATA_34 < 0) = 2;
reduced_SAMPL_DATA_41 = SAMPL_DATA_41;
reduced_SAMPL_DATA_41(reduced_SAMPL_DATA_41 > 0) = 1;
reduced_SAMPL_DATA_41(reduced_SAMPL_DATA_41 < 0) = 2;
reduced_SAMPL_DATA_42 = SAMPL_DATA_42;
reduced_SAMPL_DATA_42(reduced_SAMPL_DATA_42 > 0) = 1;
reduced_SAMPL_DATA_42(reduced_SAMPL_DATA_42 < 0) = 2;
reduced_SAMPL_DATA_43 = SAMPL_DATA_43;
reduced_SAMPL_DATA_43(reduced_SAMPL_DATA_43 > 0) = 1;
reduced_SAMPL_DATA_43(reduced_SAMPL_DATA_43 < 0) = 2;
reduced_SAMPL_DATA_44 = SAMPL_DATA_44;
reduced_SAMPL_DATA_44(reduced_SAMPL_DATA_44 > 0) = 1;
reduced_SAMPL_DATA_44(reduced_SAMPL_DATA_44 < 0) = 2;

reduced_L_SAMPL_DATA_1 = L_SAMPL_DATA_01 - 1;
reduced_L_SAMPL_DATA_2 = L_SAMPL_DATA_02 - 1;
reduced_L_SAMPL_DATA_3 = L_SAMPL_DATA_03 - 1;
reduced_L_SAMPL_DATA_4 = L_SAMPL_DATA_04 - 1;

%Sparse biopolar mapping
%creates matrix of random hypervectors with element values 1, 0, and -1,
%matrix is has feature (channel) numbers of binary D size hypervectors
%Should be the S vectors
q=0.7;

% select projM vectors
% for i = 1:1:channels_v
%     proj1_n = i;
%     proj1_p = i;
%     proj3_n = i;
%     proj3_p = i;
%     proj5_n = i;
%     proj5_p = i;
%   
%     while (proj1_n == i)
%         proj1_n = randperm(105,1);
%     end
%     while ((proj1_p == i) || (proj1_p == proj1_n))
%         proj1_p = randperm(105,1);
%     end
%     while ((proj3_n == i) || (proj3_n == proj1_n) || (proj3_n == proj1_p))
%         proj3_n = randperm(105,1);
%     end
%     while ((proj3_p == i) || (proj3_p == proj1_n) || (proj3_p == proj1_p) || (proj3_p == proj3_n))
%         proj3_p = randperm(105,1);
%     end
%     while ((proj5_n == i) || (proj5_n == proj1_n) || (proj5_n == proj1_p) || (proj5_n == proj3_n) || (proj5_n == proj3_p))
%         proj5_n = randperm(105,1);
%     end
%     while ((proj5_p == i) || (proj5_p == proj1_n) || (proj5_p == proj1_p) || (proj5_p == proj3_n) || (proj5_p == proj3_p) || (proj5_p == proj5_n))
%         proj5_p = randperm(105,1);
%     end
%     projM1_neg(i,:) = iMch5_array(proj1_n);
%     projM1_pos(i,:) = iMch5_array(proj1_p);
%     projM3_neg(i,:) = iMch5_array(proj3_n);
%     projM3_pos(i,:) = iMch5_array(proj3_p);
%     projM5_neg(i,:) = iMch5_array(proj5_n);
%     projM5_pos(i,:) = iMch5_array(proj5_p);
% end
% 
% 
% for i = channels_v+1:1:channels_v+channels_v_ECG
%     proj3_n = i;
%     proj3_p = i;
%     proj5_n = i;
%     proj5_p = i;
%     while (proj3_n == i)
%         proj3_n = randperm(105,1);
%     end
%     while ((proj3_p == i) || (proj3_p == proj3_n))
%         proj3_p = randperm(105,1);
%     end
%     while ((proj5_n == i) || (proj5_n == proj3_n) || (proj5_n == proj3_p))
%         proj5_n = randperm(105,1);
%     end
%     while ((proj5_p == i) || (proj5_p == proj3_n) || (proj5_p == proj3_p) || (proj5_p == proj5_n))
%         proj5_p = randperm(105,1);
%     end
%     projM3_neg(i,:) = iMch5_array(proj3_n);
%     projM3_pos(i,:) = iMch5_array(proj3_p);
%     projM5_neg(i,:) = iMch5_array(proj5_n);
%     projM5_pos(i,:) = iMch5_array(proj5_p);
% end  
% for i = channels_v+channels_v_ECG+1:1:channels_v+channels_v_ECG+channels_v_EEG
%     proj5_n = i;
%     proj5_p = i;
%     while (proj5_n == i)
%         proj5_n = randperm(105,1);
%     end
%     while ((proj5_p == i) || (proj5_p == proj5_n))
%         proj5_p = randperm(105,1);
%     end
%     projM5_neg(i,:) = iMch5_array(proj5_n);
%     projM5_pos(i,:) = iMch5_array(proj5_p);
% end  

for N = 3:ngram
% creates ngram for data, rotates through and 
N

%NEED TO CONVERT IMS TO MAP CONTAINERS
[x1, iMch1] = initItemMemories (D, maxL, channels_v);
[x3, iMch3] = initItemMemories (D, maxL, channels_v_EEG);
[x5, iMch5] = initItemMemories (D, maxL, channels_v_GSR);
[x7, iMch7] = initItemMemories (D, maxL, channels_v_BVP);
[x9, iMch9] = initItemMemories (D, maxL, channels_v_RES);

for i=1:1:channels_v
    iMch1(i) = iMch3_array(i,:);
end
for i=1:1:channels_v_EEG
    iMch3(i) = iMch3_array(i,:);
end
for i=1:1:channels_v_GSR
    iMch5(i) = iMch3_array(i,:);
end
for i=1:1:channels_v_BVP
    iMch7(i) = iMch3_array(i,:);
end
for i=1:1:channels_v_RES
    iMch9(i) = iMch3_array(i,:);
end

% iMch1 = containers.Map(keys(im1),iMch1);
% iMch3 = containers.Map(keys(im3),iMch3);
% iMch5 = containers.Map(keys(im5),iMch5);
% values(iMch1)

% Arousal
%generate ngram bundles for each data stream
fprintf ('HDC for L\n');
if (select == 1)
    [numpat, hdc_model_2] = hdctrainproj (classes, reduced_L_SAMPL_DATA_4, reduced_L_SAMPL_DATA_4, reduced_L_SAMPL_DATA_4, reduced_L_SAMPL_DATA_4, reduced_L_SAMPL_DATA_4,reduced_SAMPL_DATA_04, reduced_SAMPL_DATA_14, reduced_SAMPL_DATA_24, reduced_SAMPL_DATA_34, reduced_SAMPL_DATA_44, chAM14, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, channels_l, channels_l_EEG, channels_l_GSR, channels_l_BVP, channels_l_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);
else
    [numpat_2, hdc_model_2] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_2, chAM8, iMch1, D, N, precision, channels_a,projM1_pos,projM1_neg, classes); 
    [numpat_4, hdc_model_4] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_4, chAM8, iMch3, D, N, precision, channels_a_ECG,projM3_pos,projM3_neg, classes); 
    [numpat_6, hdc_model_6] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_6, chAM8, iMch5, D, N, precision, channels_a_EEG,projM5_pos,projM5_neg, classes); 
end

%bundle all the sensors (this is the fusion point)
if (select ~= 1)
    %class 1
    hdc_model_2(0)=mode([hdc_model_2(0); hdc_model_4(0); hdc_model_6(0)]);
    %class 2
    hdc_model_2(1)=mode([hdc_model_2(1); hdc_model_4(1); hdc_model_6(1)]);
end

[acc_ex2, acc2, pl2, al2, all_error] = hdcpredictproj  (reduced_L_TS_COMPLETE_4, reduced_TS_COMPLETE_04, reduced_L_TS_COMPLETE_4, reduced_TS_COMPLETE_14, reduced_L_TS_COMPLETE_4, reduced_TS_COMPLETE_24, reduced_L_TS_COMPLETE_4, reduced_TS_COMPLETE_34, reduced_L_TS_COMPLETE_4, reduced_TS_COMPLETE_44, hdc_model_2, chAM14, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, classes, channels_l, channels_l_EEG, channels_l_GSR, channels_l_BVP, channels_l_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);

accuracy(N,2) = acc2;
acc2
acc_matrix((randCounter*4),(D/1000)) = acc2;

%acc_ngram_1(N,j)=acc1;
acc_ngram_L(N,j)=acc2;

% Valence

fprintf ('HDC for D\n');
if (select == 1)
    [numpat, hdc_model_2] = hdctrainproj (classes, reduced_L_SAMPL_DATA_3, reduced_L_SAMPL_DATA_3, reduced_L_SAMPL_DATA_3, reduced_L_SAMPL_DATA_3, reduced_L_SAMPL_DATA_3,reduced_SAMPL_DATA_03, reduced_SAMPL_DATA_13, reduced_SAMPL_DATA_23, reduced_SAMPL_DATA_33, reduced_SAMPL_DATA_43, chAM14, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, channels_d, channels_d_EEG, channels_d_GSR, channels_d_BVP, channels_d_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);
else
    [numpat_2, hdc_model_2] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_2, chAM8, iMch1, D, N, precision, channels_a,projM1_pos,projM1_neg, classes); 
    [numpat_4, hdc_model_4] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_4, chAM8, iMch3, D, N, precision, channels_a_ECG,projM3_pos,projM3_neg, classes); 
    [numpat_6, hdc_model_6] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_6, chAM8, iMch5, D, N, precision, channels_a_EEG,projM5_pos,projM5_neg, classes); 
end

%bundle all the sensors (this is the fusion point)
if (select ~= 1)
    %class 1
    hdc_model_2(0)=mode([hdc_model_2(0); hdc_model_4(0); hdc_model_6(0)]);
    %class 2
    hdc_model_2(1)=mode([hdc_model_2(1); hdc_model_4(1); hdc_model_6(1)]);
end

[acc_ex2, acc2, pl2, al2, all_error] = hdcpredictproj  (reduced_L_TS_COMPLETE_3, reduced_TS_COMPLETE_03, reduced_L_TS_COMPLETE_3, reduced_TS_COMPLETE_13, reduced_L_TS_COMPLETE_3, reduced_TS_COMPLETE_23, reduced_L_TS_COMPLETE_3, reduced_TS_COMPLETE_33, reduced_L_TS_COMPLETE_3, reduced_TS_COMPLETE_43, hdc_model_2, chAM14, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, classes, channels_d, channels_d_EEG, channels_d_GSR, channels_d_BVP, channels_d_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);

accuracy(N,2) = acc2;
acc2
acc_matrix((randCounter*4 - 1),(D/1000)) = acc2;

acc_ngram_D(N,j)=acc2;

fprintf ('HDC for V\n');
if (select == 1)
    [numpat, hdc_model_2] = hdctrainproj (classes, reduced_L_SAMPL_DATA_2, reduced_L_SAMPL_DATA_2, reduced_L_SAMPL_DATA_2, reduced_L_SAMPL_DATA_2, reduced_L_SAMPL_DATA_2,reduced_SAMPL_DATA_02, reduced_SAMPL_DATA_12, reduced_SAMPL_DATA_22, reduced_SAMPL_DATA_32, reduced_SAMPL_DATA_42, chAM14, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, channels_v, channels_v_EEG, channels_v_GSR, channels_v_BVP, channels_v_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);
else
    [numpat_2, hdc_model_2] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_2, chAM8, iMch1, D, N, precision, channels_a,projM1_pos,projM1_neg, classes); 
    [numpat_4, hdc_model_4] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_4, chAM8, iMch3, D, N, precision, channels_a_ECG,projM3_pos,projM3_neg, classes); 
    [numpat_6, hdc_model_6] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_6, chAM8, iMch5, D, N, precision, channels_a_EEG,projM5_pos,projM5_neg, classes); 
end

%bundle all the sensors (this is the fusion point)
if (select ~= 1)
    %class 1
    hdc_model_2(0)=mode([hdc_model_2(0); hdc_model_4(0); hdc_model_6(0)]);
    %class 2
    hdc_model_2(1)=mode([hdc_model_2(1); hdc_model_4(1); hdc_model_6(1)]);
end

[acc_ex2, acc2, pl2, al2, all_error] = hdcpredictproj  (reduced_L_TS_COMPLETE_2, reduced_TS_COMPLETE_02, reduced_L_TS_COMPLETE_2, reduced_TS_COMPLETE_12, reduced_L_TS_COMPLETE_2, reduced_TS_COMPLETE_22, reduced_L_TS_COMPLETE_2, reduced_TS_COMPLETE_32, reduced_L_TS_COMPLETE_2, reduced_TS_COMPLETE_42, hdc_model_2, chAM14, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, classes, channels_v, channels_v_EEG, channels_v_GSR, channels_v_BVP, channels_v_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);

accuracy(N,2) = acc2;
acc2
acc_matrix((randCounter*4 - 2),(D/1000)) = acc2;

acc_ngram_V(N,j)=acc2;

fprintf ('HDC for A\n');
if (select == 1)
    [numpat, hdc_model_2] = hdctrainproj (classes, reduced_L_SAMPL_DATA_1, reduced_L_SAMPL_DATA_1, reduced_L_SAMPL_DATA_1, reduced_L_SAMPL_DATA_1, reduced_L_SAMPL_DATA_1,reduced_SAMPL_DATA_01, reduced_SAMPL_DATA_11, reduced_SAMPL_DATA_21, reduced_SAMPL_DATA_31, reduced_SAMPL_DATA_41, chAM14, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, channels_a, channels_a_EEG, channels_a_GSR, channels_a_BVP, channels_a_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);
else
    [numpat_2, hdc_model_2] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_2, chAM8, iMch1, D, N, precision, channels_a,projM1_pos,projM1_neg, classes); 
    [numpat_4, hdc_model_4] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_4, chAM8, iMch3, D, N, precision, channels_a_ECG,projM3_pos,projM3_neg, classes); 
    [numpat_6, hdc_model_6] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_6, chAM8, iMch5, D, N, precision, channels_a_EEG,projM5_pos,projM5_neg, classes); 
end

%bundle all the sensors (this is the fusion point)
if (select ~= 1)
    %class 1
    hdc_model_2(0)=mode([hdc_model_2(0); hdc_model_4(0); hdc_model_6(0)]);
    %class 2
    hdc_model_2(1)=mode([hdc_model_2(1); hdc_model_4(1); hdc_model_6(1)]);
end

[acc_ex2, acc2, pl2, al2, all_error] = hdcpredictproj  (reduced_L_TS_COMPLETE_1, reduced_TS_COMPLETE_01, reduced_L_TS_COMPLETE_1, reduced_TS_COMPLETE_11, reduced_L_TS_COMPLETE_1, reduced_TS_COMPLETE_21, reduced_L_TS_COMPLETE_1, reduced_TS_COMPLETE_31, reduced_L_TS_COMPLETE_1, reduced_TS_COMPLETE_41, hdc_model_2, chAM14, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, classes, channels_a, channels_a_EEG, channels_a_GSR, channels_a_BVP, channels_a_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);

accuracy(N,2) = acc2;
acc2
acc_matrix((randCounter*4 - 3),(D/1000)) = acc2;

%acc_ngram_1(N,j)=acc1;
acc_ngram_A(N,j)=acc2;
end

for i = 1:1:length(1:ngram)
    if acc_ngram_A(i,j) > 0.684
        met_A_accuracy(j) = 1;
    end
    if acc_ngram_V(i,j) > 0.801
        met_V_accuracy(j) = 1;
    end
    if acc_ngram_D(i,j) > 0.7
        met_D_accuracy(j) = 1;
    end
    if acc_ngram_L(i,j) > 0.7
        met_L_accuracy(j) = 1;
    end
    
end

end

iMfull = [];
for i = 1:1:iMch9.Count
    iMfull = [iMfull iMch9(i)]; %#ok<AGROW>
end

projM_pos_full = [];
%projM_pos_temp = [projM1_pos; projM3_pos; projM5_pos];
x = size(projM9_pos);
dim = x(1);
for i = 1:1:dim
    projM_pos_full = [projM_pos_full projM9_pos(i,:)]; %#ok<AGROW>
end

projM_neg_full = [];
%projM_neg_temp = [projM1_neg; projM3_neg; projM5_neg];
x = size(projM9_neg);
dim = x(1);
for i = 1:1:dim
    projM_neg_full = [projM_neg_full projM9_neg(i,:)]; %#ok<AGROW>
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
