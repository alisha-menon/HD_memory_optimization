clear;

% set for simulation
D_full = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]; %dimension of the hypervectors
repetitions = 10;

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
features_ECG=features(:,1+32:32+77); 
features_EEG=features(:,1+32+77:32+77+105); 

%=======HDC============
% HD_functions: uncommented, original functions (has bug)
% HD_functions_commented: late fusion w/comments and w/bug fix 
% HD_functions_mod_reduced: early fusion w/bug fix and data as {0, 1, 2}
    % instead of original feature values
% HD_functions_modified: early fusion w/bug fix
% HD_functions_modified_64permute: early fusion with permutation operation adds a 0 every 64 bits
% HD_functions_multiplex: late fusion with multiplexer for projM
% HD_functions_multiplex_64permute: late fusion w/bug fix + multiplexed spatial encoding, permutate modified to logical shift right and replace 0 every 64 bits for vectorized, reduced dataset
% HD_functions_parameter_less_tempenc: HD early fusion w/parameter-less temporal encoder
% HD_functions_train_parameterless: HD early fusion w/parameter-less
    % temporal encoder training
% HD_functions_train_parameterless_bundletest: HD early fusion w/parameter-less
    % temporal encoder training some experiments honestly I forgot and
    % probably irrelevent, don't need any of the parameterless stuff most likely...

%% choose select
% select = 1 for early fusion
% select = 2 for late fusion
select = 1;
if (select == 1)
    HD_functions_iM_repeatmod;     % load HD functions
else 
    HD_functions_multiplex;
end

acc_ngram_1=[];
acc_ngram_2=[];


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

learningrate=0.25; % percentage of the dataset used to train the algorithm
q=0.5;
accuracy_A = zeros(repetitions,length(D_full));
accuracy_V = zeros(repetitions,length(D_full));

learningFrac = learningrate(1); 
learningFrac;
classes = 2; % level of classes
precision = 20; %no use
ngram = 3; % for temporal encode
N = ngram;
maxL = 2; % for IM gen
 
channels_v_EXG=channels_v +channels_v_ECG+channels_v_EEG;
channels_a_EXG=channels_a+channels_a_ECG+channels_a_EEG;

%downsample the dataset using the value contained in the variable "downSampRate"
%returns downsampled data which skips every 8 of the original dataset
downSampRate = 8;
LABEL_1_v=f_label_v_binary;
LABEL_1_a=f_label_a_binary;
[TS_COMPLETE_1, L_TS_COMPLETE_1] = downSampling (COMPLETE_1_v, LABEL_1_v, downSampRate);
[TS_COMPLETE_2, L_TS_COMPLETE_2] = downSampling (COMPLETE_1_a, LABEL_1_a, downSampRate);
[TS_COMPLETE_3, L_TS_COMPLETE_3] = downSampling (COMPLETE_1_v_ECG, LABEL_1_v, downSampRate);
[TS_COMPLETE_4, L_TS_COMPLETE_4] = downSampling (COMPLETE_1_a_ECG, LABEL_1_a, downSampRate);
[TS_COMPLETE_5, L_TS_COMPLETE_5] = downSampling (COMPLETE_1_v_EEG, LABEL_1_v, downSampRate);
[TS_COMPLETE_6, L_TS_COMPLETE_6] = downSampling (COMPLETE_1_a_EEG, LABEL_1_a, downSampRate);
reduced_TS_COMPLETE_1 = TS_COMPLETE_1;
reduced_TS_COMPLETE_1(reduced_TS_COMPLETE_1 > 0) = 1;
reduced_TS_COMPLETE_1(reduced_TS_COMPLETE_1 < 0) = 2;
reduced_TS_COMPLETE_2 = TS_COMPLETE_2;
reduced_TS_COMPLETE_2(reduced_TS_COMPLETE_2 > 0) = 1;
reduced_TS_COMPLETE_2(reduced_TS_COMPLETE_2 < 0) = 2;
reduced_TS_COMPLETE_3 = TS_COMPLETE_3;
reduced_TS_COMPLETE_3(reduced_TS_COMPLETE_3 > 0) = 1;
reduced_TS_COMPLETE_3(reduced_TS_COMPLETE_3 < 0) = 2;
reduced_TS_COMPLETE_4 = TS_COMPLETE_4;
reduced_TS_COMPLETE_4(reduced_TS_COMPLETE_4 > 0) = 1;
reduced_TS_COMPLETE_4(reduced_TS_COMPLETE_4 < 0) = 2;
reduced_TS_COMPLETE_5 = TS_COMPLETE_5;
reduced_TS_COMPLETE_5(reduced_TS_COMPLETE_5 > 0) = 1;
reduced_TS_COMPLETE_5(reduced_TS_COMPLETE_5 < 0) = 2;
reduced_TS_COMPLETE_6 = TS_COMPLETE_6;
reduced_TS_COMPLETE_6(reduced_TS_COMPLETE_6 > 0) = 1;
reduced_TS_COMPLETE_6(reduced_TS_COMPLETE_6 < 0) = 2;
reduced_L_TS_COMPLETE_1 = L_TS_COMPLETE_1;
reduced_L_TS_COMPLETE_1(reduced_L_TS_COMPLETE_1 == 1) = 0;
reduced_L_TS_COMPLETE_1(reduced_L_TS_COMPLETE_1 == 2) = 1;
reduced_L_TS_COMPLETE_2 = L_TS_COMPLETE_2;
reduced_L_TS_COMPLETE_2(reduced_L_TS_COMPLETE_2 == 1) = 0;
reduced_L_TS_COMPLETE_2(reduced_L_TS_COMPLETE_2 == 2) = 1;

% %Valence
% valence_count_class_change = 0;
% for i = 1:1:length(LABEL_1_v)-1
%     if LABEL_1_v(i) ~= LABEL_1_v(i+1)
%         valence_count_class_change = valence_count_class_change+1;
%     end
% end
% %arousal
% arousal_count_class_change = 0;
% for i = 1:1:length(LABEL_1_a)-1
%     if LABEL_1_a(i) ~= LABEL_1_a(i+1)
%         arousal_count_class_change = arousal_count_class_change+1;
%     end
% end
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
    [L1_1, L2_1, L_SAMPL_DATA_1, SAMPL_DATA_1, L_SAMPL_DATA_1_test, SAMPL_DATA_1_test] = genTrainTestData (TS_COMPLETE_1, L_TS_COMPLETE_1, learningFrac, 'inorder',N);
    [L1_2, L2_2, L_SAMPL_DATA_2, SAMPL_DATA_2, L_SAMPL_DATA_2_test, SAMPL_DATA_2_test] = genTrainTestData (TS_COMPLETE_2, L_TS_COMPLETE_2, learningFrac, 'inorder',N);
    [SAMPL_DATA_3, SAMPL_DATA_3_test] = select_traintest(L1_1, L2_1, TS_COMPLETE_3);
    [SAMPL_DATA_4, SAMPL_DATA_4_test] = select_traintest(L1_2, L2_2, TS_COMPLETE_4);
    [SAMPL_DATA_5, SAMPL_DATA_5_test] = select_traintest(L1_1, L2_1, TS_COMPLETE_5);
    [SAMPL_DATA_6, SAMPL_DATA_6_test] = select_traintest(L1_2, L2_2, TS_COMPLETE_6);

    reduced_SAMPL_DATA_1 = SAMPL_DATA_1;
    reduced_SAMPL_DATA_1(reduced_SAMPL_DATA_1 > 0) = 1;
    reduced_SAMPL_DATA_1(reduced_SAMPL_DATA_1 < 0) = 2;
    reduced_SAMPL_DATA_1_test = SAMPL_DATA_1_test;
    reduced_SAMPL_DATA_1_test(reduced_SAMPL_DATA_1_test > 0) = 1;
    reduced_SAMPL_DATA_1_test(reduced_SAMPL_DATA_1_test < 0) = 2;
    reduced_SAMPL_DATA_2 = SAMPL_DATA_2;
    reduced_SAMPL_DATA_2(reduced_SAMPL_DATA_2 > 0) = 1;
    reduced_SAMPL_DATA_2(reduced_SAMPL_DATA_2 < 0) = 2;
    reduced_SAMPL_DATA_2_test = SAMPL_DATA_2_test;
    reduced_SAMPL_DATA_2_test(reduced_SAMPL_DATA_2_test > 0) = 1;
    reduced_SAMPL_DATA_2_test(reduced_SAMPL_DATA_2_test < 0) = 2;
    reduced_SAMPL_DATA_3 = SAMPL_DATA_3;
    reduced_SAMPL_DATA_3(reduced_SAMPL_DATA_3 > 0) = 1;
    reduced_SAMPL_DATA_3(reduced_SAMPL_DATA_3 < 0) = 2;
    reduced_SAMPL_DATA_3_test = SAMPL_DATA_3_test;
    reduced_SAMPL_DATA_3_test(reduced_SAMPL_DATA_3_test > 0) = 1;
    reduced_SAMPL_DATA_3_test(reduced_SAMPL_DATA_3_test < 0) = 2;
    reduced_SAMPL_DATA_4 = SAMPL_DATA_4;
    reduced_SAMPL_DATA_4(reduced_SAMPL_DATA_4 > 0) = 1;
    reduced_SAMPL_DATA_4(reduced_SAMPL_DATA_4 < 0) = 2;
    reduced_SAMPL_DATA_4_test = SAMPL_DATA_4_test;
    reduced_SAMPL_DATA_4_test(reduced_SAMPL_DATA_4_test > 0) = 1;
    reduced_SAMPL_DATA_4_test(reduced_SAMPL_DATA_4_test < 0) = 2;
    reduced_SAMPL_DATA_5 = SAMPL_DATA_5;
    reduced_SAMPL_DATA_5(reduced_SAMPL_DATA_5 > 0) = 1;
    reduced_SAMPL_DATA_5(reduced_SAMPL_DATA_5 < 0) = 2;
    reduced_SAMPL_DATA_5_test = SAMPL_DATA_5_test;
    reduced_SAMPL_DATA_5_test(reduced_SAMPL_DATA_5_test > 0) = 1;
    reduced_SAMPL_DATA_5_test(reduced_SAMPL_DATA_5_test < 0) = 2;
    reduced_SAMPL_DATA_6 = SAMPL_DATA_6;
    reduced_SAMPL_DATA_6(reduced_SAMPL_DATA_6 > 0) = 1;
    reduced_SAMPL_DATA_6(reduced_SAMPL_DATA_6 < 0) = 2;
    reduced_SAMPL_DATA_6_test = SAMPL_DATA_6_test;
    reduced_SAMPL_DATA_6_test(reduced_SAMPL_DATA_6_test > 0) = 1;
    reduced_SAMPL_DATA_6_test(reduced_SAMPL_DATA_6_test < 0) = 2;
    reduced_L_SAMPL_DATA_1 = L_SAMPL_DATA_1;
    reduced_L_SAMPL_DATA_1(reduced_L_SAMPL_DATA_1 == 1) = 0;
    reduced_L_SAMPL_DATA_1(reduced_L_SAMPL_DATA_1 == 2) = 1;
    reduced_L_SAMPL_DATA_1_test = L_SAMPL_DATA_1_test;
    reduced_L_SAMPL_DATA_1_test(reduced_L_SAMPL_DATA_1_test == 1) = 0;
    reduced_L_SAMPL_DATA_1_test(reduced_L_SAMPL_DATA_1_test == 2) = 1;
    reduced_L_SAMPL_DATA_2 = L_SAMPL_DATA_2;
    reduced_L_SAMPL_DATA_2(reduced_L_SAMPL_DATA_2 == 1) = 0;
    reduced_L_SAMPL_DATA_2(reduced_L_SAMPL_DATA_2 == 2) = 1;
    reduced_L_SAMPL_DATA_2_test = L_SAMPL_DATA_2_test;
    reduced_L_SAMPL_DATA_2_test(reduced_L_SAMPL_DATA_2_test == 1) = 0;
    reduced_L_SAMPL_DATA_2_test(reduced_L_SAMPL_DATA_2_test == 2) = 1;

for j=1:repetitions
    j
    
    for dim_loop = 1:1:length(D_full)
        D = D_full(dim_loop);
        rng('shuffle');
        [chAM1, iMch1] = initItemMemories (D, maxL, channels_v);
        %[chAM2, iMch2] = initItemMemories (D, maxL, channels_a);
        [chAM3, iMch3] = initItemMemories (D, maxL, channels_v_ECG);
        %[chAM4, iMch4] = initItemMemories (D, maxL, channels_a_ECG);
        [chAM5, iMch5] = initItemMemories (D, maxL, channels_v_EEG);
        %[chAM6, iMch6] = initItemMemories (D, maxL, channels_a_EEG);
        [chAM7, iMch7] = initItemMemories (D, maxL, channels_v_EXG);
        [chAM8, iMch8] = initItemMemories (D, maxL, channels_a_EXG);
        projM1_neg=projRandomHV(D,channels_v);
        projM1_pos=projRandomHV(D,channels_v);
        projM2_neg=projRandomHV(D,channels_a);
        projM2_pos=projRandomHV(D,channels_a);
        projM3_neg=projRandomHV(D,channels_v_ECG);
        projM3_pos=projRandomHV(D,channels_v_ECG);
        projM4_neg=projRandomHV(D,channels_a_ECG);
        projM4_pos=projRandomHV(D,channels_a_ECG);
        projM5_neg=projRandomHV(D,channels_v_EEG);
        projM5_pos=projRandomHV(D,channels_v_EEG);
        projM6_neg=projRandomHV(D,channels_a_EEG);
        projM6_pos=projRandomHV(D,channels_a_EEG);
        % Arousal
        %generate ngram bundles for each data stream
        fprintf ('HDC for A\n');
        if (select == 1)
            [numpat, hdc_model_2] = hdctrainproj (classes, reduced_L_SAMPL_DATA_2, reduced_L_SAMPL_DATA_2, reduced_L_SAMPL_DATA_2,reduced_SAMPL_DATA_2, reduced_SAMPL_DATA_4, reduced_SAMPL_DATA_6, chAM8, iMch1, iMch3, iMch5, D, N, precision, channels_a, channels_a_ECG, channels_a_EEG,projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg); 
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

        [acc_ex2, acc2, pl2, al2, all_error] = hdcpredictproj  (reduced_L_SAMPL_DATA_2_test, reduced_SAMPL_DATA_2_test, reduced_L_SAMPL_DATA_2_test, reduced_SAMPL_DATA_4_test, reduced_L_SAMPL_DATA_2_test, reduced_SAMPL_DATA_6_test,hdc_model_2, chAM8, iMch1, iMch3, iMch5, D, N, precision, classes, channels_a,channels_a_ECG,channels_a_EEG, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg);

        accuracy_A(j,dim_loop) = acc2;
        acc2

        %acc_ngram_1(N,j)=acc1;
        %acc_ngram_A(N,j)=acc2;

        % Valence

        fprintf ('HDC for V\n');
        if (select == 1)
             [numpat, hdc_model_1] = hdctrainproj (classes, reduced_L_SAMPL_DATA_1, reduced_L_SAMPL_DATA_1, reduced_L_SAMPL_DATA_1,reduced_SAMPL_DATA_1, reduced_SAMPL_DATA_3, reduced_SAMPL_DATA_5, chAM8, iMch1, iMch3, iMch5, D, N, precision, channels_v, channels_v_ECG, channels_v_EEG,projM1_pos,projM1_neg, projM3_pos,projM3_neg, projM5_pos,projM5_neg); 
        else
            [numpat_1, hdc_model_1] = hdctrainproj (reduced_L_SAMPL_DATA_1, reduced_SAMPL_DATA_1, chAM8, iMch1, D, N, precision, channels_v,projM1_pos,projM1_neg, classes); 
            [numpat_3, hdc_model_3] = hdctrainproj (reduced_L_SAMPL_DATA_1, reduced_SAMPL_DATA_3, chAM8, iMch3, D, N, precision, channels_v_ECG,projM3_pos,projM3_neg, classes); 
            [numpat_5, hdc_model_5] = hdctrainproj (reduced_L_SAMPL_DATA_1, reduced_SAMPL_DATA_5, chAM8, iMch5, D, N, precision, channels_v_EEG,projM5_pos,projM5_neg, classes); 
        end

        if (select ~= 1)
            %class 1
            hdc_model_1(0)=mode([hdc_model_1(0); hdc_model_3(0); hdc_model_5(0)]);
            %class 2
            hdc_model_1(1)=mode([hdc_model_1(1); hdc_model_3(1); hdc_model_5(1)]);
        end

        [acc_ex1, acc1, pl1, al1, all_error] = hdcpredictproj  (reduced_L_SAMPL_DATA_1_test, reduced_SAMPL_DATA_1_test, reduced_L_SAMPL_DATA_1_test, reduced_SAMPL_DATA_3_test, reduced_L_SAMPL_DATA_1_test, reduced_SAMPL_DATA_5_test,hdc_model_1, chAM8, iMch1, iMch3, iMch5, D, N, precision, classes, channels_v,channels_v_ECG,channels_v_EEG,projM1_pos,projM1_neg,projM3_pos,projM3_neg,projM5_pos,projM5_neg);
        %for verification
        %[acc_ex1, acc1, pl1, al1, all_error] = hdcpredictproj  (L_SAMPL_DATA_1, SAMPL_DATA_1, L_SAMPL_DATA_3, SAMPL_DATA_3, L_SAMPL_DATA_5, SAMPL_DATA_5,hdc_model, chAM8, iMch1, iMch3, iMch5, D, N, precision, classes, channels_v,channels_v_ECG,channels_v_EEG,projM1,projM3,projM5);

        accuracy_V(j,dim_loop) = acc1;
        acc1

        %acc_ngram_1(N,j)=acc1;
        %acc_ngram_V(N,j)=acc1;
    end

end

aaaccuracy_A = mean(accuracy_A);
aaaccuracy_V = mean(accuracy_V);

% iMfull = [];
% for i = 1:1:iMch7.Count
%     iMfull = [iMfull iMch7(i)]; %#ok<AGROW>
% end
% 
% projM_pos_full = [];
% projM_pos_temp = [projM1_pos; projM3_pos; projM5_pos];
% x = size(projM_pos_temp);
% dim = x(1);
% for i = 1:1:dim
%     projM_pos_full = [projM_pos_full projM_pos_temp(i,:)]; %#ok<AGROW>
% end
% 
% projM_neg_full = [];
% projM_neg_temp = [projM1_neg; projM3_neg; projM5_neg];
% x = size(projM_neg_temp);
% dim = x(1);
% for i = 1:1:dim
%     projM_neg_full = [projM_neg_full projM_neg_temp(i,:)]; %#ok<AGROW>
% end

