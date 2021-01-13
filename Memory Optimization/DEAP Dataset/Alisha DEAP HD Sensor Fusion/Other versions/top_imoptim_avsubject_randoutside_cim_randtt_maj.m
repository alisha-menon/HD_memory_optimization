clear;

%% choose select
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
    
% select = 1 for early fusion
% select = 2 for late fusion
select = 1;
if (select == 1)
    HD_functions_avsubject_cim_randtt_maj;     % load HD functions
else 
    HD_functions_multiplex;
end

%%
randCounter= 10; %per subject
full_count = randCounter;
learningrate=0.5; % percentage of the dataset used to train the algorithm
downSampRate = 1;
ngram = 3; % for temporal encode
subjects = 32;
% D_full = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]; %dimension of the hypervectors
D_full = [10000];
maxL = 10; % for CiM
N = ngram;
precision = 10; % for CiM
%====Features and Label===
load('DEAP_data.mat')
features=inputs(:,:);
features_EMG=features(:,1:10);
features_EEG=features(:,11:202);
features_GSR=features(:,203:209);
features_BVP=features(:,210:226);
features_RES=features(:,227:238);
channels_v=length(features_EMG(1,:));
channels_v_EEG=length(features_EEG(1,:));
channels_v_GSR=length(features_GSR(1,:));
channels_v_BVP=length(features_BVP(1,:));
channels_v_RES=length(features_RES(1,:));
channels_v_EXG=channels_v+channels_v_EEG+channels_v_GSR+channels_v_BVP+channels_v_RES;

acc_matrix = zeros(randCounter*2,subjects);
acc_matrix_ex = zeros(randCounter*2,subjects);
acc_matrix_ex_all = zeros(randCounter*2,subjects);
acc_matrix_A = zeros(randCounter,subjects);
acc_matrix_V = zeros(randCounter,subjects);
acc_matrix_A_mod1 = zeros(randCounter,subjects);
acc_matrix_V_mod1 = zeros(randCounter,subjects);
acc_matrix_A_mod2 = zeros(randCounter,subjects);
acc_matrix_V_mod2 = zeros(randCounter,subjects);
acc_matrix_A_mod3 = zeros(randCounter,subjects);
acc_matrix_V_mod3 = zeros(randCounter,subjects);
acc_matrix_A_mod4 = zeros(randCounter,subjects);
acc_matrix_V_mod4 = zeros(randCounter,subjects);
acc_matrix_A_mod5 = zeros(randCounter,subjects);
acc_matrix_V_mod5 = zeros(randCounter,subjects);
acc_matrix_A_majcount = zeros(randCounter,subjects);
acc_matrix_V_majcount = zeros(randCounter,subjects);

acc_matrix_Vex = zeros(randCounter,subjects);
acc_matrix_Aex = zeros(randCounter,subjects);
acc_matrix_Vex_all = zeros(randCounter,subjects);
acc_matrix_Aex_all = zeros(randCounter,subjects);
acc_matrix_Vex_all_mod1 = zeros(randCounter,subjects);
acc_matrix_Aex_all_mod1 = zeros(randCounter,subjects);
acc_matrix_Vex_all_mod2 = zeros(randCounter,subjects);
acc_matrix_Aex_all_mod2 = zeros(randCounter,subjects);
acc_matrix_Vex_all_mod3 = zeros(randCounter,subjects);
acc_matrix_Aex_all_mod3 = zeros(randCounter,subjects);
acc_matrix_Vex_all_mod4 = zeros(randCounter,subjects);
acc_matrix_Aex_all_mod4 = zeros(randCounter,subjects);
acc_matrix_Vex_all_mod5 = zeros(randCounter,subjects);
acc_matrix_Aex_all_mod5 = zeros(randCounter,subjects);
acc_matrix_Vex_all_majcount = zeros(randCounter,subjects);
acc_matrix_Aex_all_majcount = zeros(randCounter,subjects);

D = D_full;
q=0.7;
learningFrac = learningrate(1); 
while (randCounter>0)
    rng('shuffle');
    
    [chAM1, iMch1] = initItemMemories (D, maxL, channels_v);
    [chAM3, iMch3] = initItemMemories (D, maxL, channels_v_EEG);
    [chAM5, iMch5] = initItemMemories (D, maxL, channels_v_GSR);
    [chAM7, iMch7] = initItemMemories (D, maxL, channels_v_BVP);
    [chAM9, iMch9] = initItemMemories (D, maxL, channels_v_RES);
    [chAM8, iMch8] = initItemMemories (D, maxL, channels_v_EXG);
    %Sparse biopolar mapping
    %creates matrix of random hypervectors with element values 1, 0, and -1,
    %matrix is has feature (channel) numbers of binary D size hypervectors
    %Should be the S vectors
%     projM1=projBRandomHV(D,channels_v,q);
%     projM3=projBRandomHV(D,channels_v_EEG,q);
%     projM5=projBRandomHV(D,channels_v_GSR,q);
%     projM7=projBRandomHV(D,channels_v_BVP,q);
%     projM9=projBRandomHV(D,channels_v_RES,q);
    
    for subject = 1:1:subjects
        subject
        features=inputs((subject-1)*40+1:subject*40,:);
        f_label_a_binary=features(:,239);
        f_label_v_binary=features(:,240);

        m = 5;

        % f_label = f_label_a_binary;
        % f_label(f_label <= 3) = 1;
        % f_label(f_label >= 7) = 2;
        % f_label_ind = f_label <=2;
        % size = sum(f_label_ind);
        % 
        % f_label_a_binary = f_label(f_label <=2);
        % f_label_v_binary = f_label(f_label <=2);
        % f_label_d_binary = f_label(f_label <=2);
        % f_label_l_binary = f_label(f_label <=2);


        f_label_a_binary(f_label_a_binary < m) = 1;
        f_label_a_binary(f_label_a_binary >= m) = 2;
        f_label_v_binary(f_label_v_binary < m) = 1;
        f_label_v_binary(f_label_v_binary >= m) = 2;

        for k=1:238
         features(:,k)=features(:,k)-min(features(:,k));
        end

        for k=1:238
         features(:,k)=features(:,k)/max(features(:,k));
        end

%         for i=1:238
%          features(:,i)=2*features(:,i)-1;
%         end
% 
%         for i=1:238
%          features(:,i)=features(:,i)+0.35;
%         end

        features_EMG=features(:,1:10);
        features_EEG=features(:,11:202);
        features_GSR=features(:,203:209);
        features_BVP=features(:,210:226);
        features_RES=features(:,227:238);

        channels_a=channels_v;
        channels_a_EEG=channels_v_EEG;
        channels_a_GSR=channels_v_GSR;
        channels_a_BVP=channels_v_BVP;
        channels_a_RES=channels_v_RES;

        COMPLETE_1_v=features_EMG;
        COMPLETE_1_a=features_EMG;

        COMPLETE_1_v_EEG=features_EEG;
        COMPLETE_1_a_EEG=features_EEG;

        COMPLETE_1_v_GSR=features_GSR;
        COMPLETE_1_a_GSR=features_GSR;

        COMPLETE_1_v_BVP=features_BVP;
        COMPLETE_1_a_BVP=features_BVP;

        COMPLETE_1_v_RES=features_RES;
        COMPLETE_1_a_RES=features_RES;

        classes = 2; % level of classes
        channels_a_EXG=channels_a+channels_a_EEG+channels_a_GSR+channels_a_BVP+channels_a_RES;

        % iMch1 = containers.Map ('KeyType','double','ValueType','any');
        % iMch3 = containers.Map ('KeyType','double','ValueType','any');
        % iMch5 = containers.Map ('KeyType','double','ValueType','any');
        % iMch7 = containers.Map ('KeyType','double','ValueType','any');
        % iMch9 = containers.Map ('KeyType','double','ValueType','any');
        % 
        % total_vectors = max([channels_v_EEG, channels_v_GSR, channels_v, channels_v_BVP, channels_v_RES]);
        % k = ceil(log2(total_vectors));
        % binary = fliplr(flip(dec2bin(2^k-1:-1:0)-'0'));
        % 
        % seed = randCounter;
        % while seed>0
        %     seedVectors = [genRandomHV(D);genRandomHV(D)];
        %     seed = seed - 1;
        % end
        % iMvectors  = containers.Map ('KeyType','double','ValueType','any');
        % for i = 1:total_vectors
        %     vec = seedVectors(1+binary(i,1), :);
        %     for d = 2:length(binary(i,:))
        %         vec = xor(vec, circshift(seedVectors(1+binary(i,d), :),d-1));
        %     end
        %     iMvectors(i) = vec;
        % end
        % 
        % for i=1:1:channels_v_EEG
        %     iMch3(i) = iMvectors(i);
        % end
        % for i=1:1:channels_v_GSR
        %     iMch5(i) = iMch3(i);
        % end
        % for i=1:1:channels_v
        %     iMch1(i) = iMch3(i);
        % end
        % for i=1:1:channels_v_BVP
        %     iMch7(i) = iMch3(i);
        % end
        % for i=1:1:channels_v_RES
        %     iMch9(i) = iMch3(i);
        % end

        % [chAM7, iMch7] = initItemMemories (D, maxL, channels_v_EXG);

        %downsample the dataset using the value contained in the variable "downSampRate"
        %returns downsampled data which skips every 8 of the original dataset
        LABEL_1_v=f_label_v_binary;
        LABEL_1_a=f_label_a_binary;

        [TS_COMPLETE_01, L_TS_COMPLETE_01] = downSampling (COMPLETE_1_v, LABEL_1_v, downSampRate); %emg
        [TS_COMPLETE_02, L_TS_COMPLETE_02] = downSampling (COMPLETE_1_a, LABEL_1_a, downSampRate);

        [TS_COMPLETE_11, L_TS_COMPLETE_11] = downSampling (COMPLETE_1_v_EEG, LABEL_1_v, downSampRate);
        [TS_COMPLETE_12, L_TS_COMPLETE_12] = downSampling (COMPLETE_1_a_EEG, LABEL_1_a, downSampRate);

        [TS_COMPLETE_21, L_TS_COMPLETE_21] = downSampling (COMPLETE_1_v_GSR, LABEL_1_v, downSampRate);
        [TS_COMPLETE_22, L_TS_COMPLETE_22] = downSampling (COMPLETE_1_a_GSR, LABEL_1_a, downSampRate);

        [TS_COMPLETE_31, L_TS_COMPLETE_31] = downSampling (COMPLETE_1_v_BVP, LABEL_1_v, downSampRate);
        [TS_COMPLETE_32, L_TS_COMPLETE_32] = downSampling (COMPLETE_1_a_BVP, LABEL_1_a, downSampRate);


        [TS_COMPLETE_41, L_TS_COMPLETE_41] = downSampling (COMPLETE_1_v_RES, LABEL_1_v, downSampRate);
        [TS_COMPLETE_42, L_TS_COMPLETE_42] = downSampling (COMPLETE_1_a_RES, LABEL_1_a, downSampRate);


        reduced_TS_COMPLETE_01 = TS_COMPLETE_01;
%         reduced_TS_COMPLETE_01(reduced_TS_COMPLETE_01 > 0) = 1;
%         reduced_TS_COMPLETE_01(reduced_TS_COMPLETE_01 < 0) = 2;
        reduced_TS_COMPLETE_02 = TS_COMPLETE_02;
%         reduced_TS_COMPLETE_02(reduced_TS_COMPLETE_02 > 0) = 1;
%         reduced_TS_COMPLETE_02(reduced_TS_COMPLETE_02 < 0) = 2;
        reduced_TS_COMPLETE_11 = TS_COMPLETE_11;
%         reduced_TS_COMPLETE_11(reduced_TS_COMPLETE_11 > 0) = 1;
%         reduced_TS_COMPLETE_11(reduced_TS_COMPLETE_11 < 0) = 2;
        reduced_TS_COMPLETE_12 = TS_COMPLETE_12;
%         reduced_TS_COMPLETE_12(reduced_TS_COMPLETE_12 > 0) = 1;
%         reduced_TS_COMPLETE_12(reduced_TS_COMPLETE_12 < 0) = 2;
        reduced_TS_COMPLETE_21 = TS_COMPLETE_21;
%         reduced_TS_COMPLETE_21(reduced_TS_COMPLETE_21 > 0) = 1;
%         reduced_TS_COMPLETE_21(reduced_TS_COMPLETE_21 < 0) = 2;
        reduced_TS_COMPLETE_22 = TS_COMPLETE_22;
%         reduced_TS_COMPLETE_22(reduced_TS_COMPLETE_22 > 0) = 1;
%         reduced_TS_COMPLETE_22(reduced_TS_COMPLETE_22 < 0) = 2;
        reduced_TS_COMPLETE_31 = TS_COMPLETE_31;
%         reduced_TS_COMPLETE_31(reduced_TS_COMPLETE_31 > 0) = 1;
%         reduced_TS_COMPLETE_31(reduced_TS_COMPLETE_31 < 0) = 2;
        reduced_TS_COMPLETE_32 = TS_COMPLETE_32;
%         reduced_TS_COMPLETE_32(reduced_TS_COMPLETE_32 > 0) = 1;
%         reduced_TS_COMPLETE_32(reduced_TS_COMPLETE_32 < 0) = 2;
        reduced_TS_COMPLETE_41 = TS_COMPLETE_41;
%         reduced_TS_COMPLETE_41(reduced_TS_COMPLETE_41 > 0) = 1;
%         reduced_TS_COMPLETE_41(reduced_TS_COMPLETE_41 < 0) = 2;
        reduced_TS_COMPLETE_42 = TS_COMPLETE_42;
%         reduced_TS_COMPLETE_42(reduced_TS_COMPLETE_42 > 0) = 1;
%         reduced_TS_COMPLETE_42(reduced_TS_COMPLETE_42 < 0) = 2;
        reduced_L_TS_COMPLETE_1 = L_TS_COMPLETE_01;
        reduced_L_TS_COMPLETE_1(reduced_L_TS_COMPLETE_1 == 1) = 0;
        reduced_L_TS_COMPLETE_1(reduced_L_TS_COMPLETE_1 == 2) = 1;
        reduced_L_TS_COMPLETE_2 = L_TS_COMPLETE_02;
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
        % gen training data finds all the samples corresponding to labels up to 7
        % (only see 1 and 2 in the data though). It allocates a certain percentage
        % to training data. Then it creates a dataset with labels corresponding to
        % the selected data for training. The label dataset is in order from 1-7
        % and the data is also stacked one by one so that it is in order from 1-7

        [L1_1, L2_1, L_SAMPL_DATA_01_train, SAMPL_DATA_01_train, L_SAMPL_DATA_01_test, SAMPL_DATA_01_test] = genTrainTestData (TS_COMPLETE_01, L_TS_COMPLETE_01, learningFrac, 'inorder');
        [L1_2, L2_2, L_SAMPL_DATA_02_train, SAMPL_DATA_02_train, L_SAMPL_DATA_02_test, SAMPL_DATA_02_test] = genTrainTestData (TS_COMPLETE_02, L_TS_COMPLETE_02, learningFrac, 'inorder');
        [SAMPL_DATA_11_train, SAMPL_DATA_11_test] = select_traintest(L1_1, L2_1, TS_COMPLETE_11);
        [SAMPL_DATA_12_train, SAMPL_DATA_12_test] = select_traintest(L1_2, L2_2, TS_COMPLETE_12);
        [SAMPL_DATA_21_train, SAMPL_DATA_21_test] = select_traintest(L1_1, L2_1, TS_COMPLETE_21);
        [SAMPL_DATA_22_train, SAMPL_DATA_22_test] = select_traintest(L1_2, L2_2, TS_COMPLETE_22);
        [SAMPL_DATA_31_train, SAMPL_DATA_31_test] = select_traintest(L1_1, L2_1, TS_COMPLETE_31);
        [SAMPL_DATA_32_train, SAMPL_DATA_32_test] = select_traintest(L1_2, L2_2, TS_COMPLETE_32);
        [SAMPL_DATA_41_train, SAMPL_DATA_41_test] = select_traintest(L1_1, L2_1, TS_COMPLETE_41);
        [SAMPL_DATA_42_train, SAMPL_DATA_42_test] = select_traintest(L1_2, L2_2, TS_COMPLETE_42);

        reduced_SAMPL_DATA_01_train = SAMPL_DATA_01_train;
%         reduced_SAMPL_DATA_01_train(reduced_SAMPL_DATA_01_train > 0) = 1;
%         reduced_SAMPL_DATA_01_train(reduced_SAMPL_DATA_01_train < 0) = 2;
        reduced_SAMPL_DATA_02_train = SAMPL_DATA_02_train;
%         reduced_SAMPL_DATA_02_train(reduced_SAMPL_DATA_02_train > 0) = 1;
%         reduced_SAMPL_DATA_02_train(reduced_SAMPL_DATA_02_train < 0) = 2;
        reduced_SAMPL_DATA_01_test = SAMPL_DATA_01_test;
%         reduced_SAMPL_DATA_01_test(reduced_SAMPL_DATA_01_test > 0) = 1;
%         reduced_SAMPL_DATA_01_test(reduced_SAMPL_DATA_01_test < 0) = 2;
        reduced_SAMPL_DATA_02_test = SAMPL_DATA_02_test;
%         reduced_SAMPL_DATA_02_test(reduced_SAMPL_DATA_02_test > 0) = 1;
%         reduced_SAMPL_DATA_02_test(reduced_SAMPL_DATA_02_test < 0) = 2;
        reduced_SAMPL_DATA_11_train = SAMPL_DATA_11_train;
%         reduced_SAMPL_DATA_11_train(reduced_SAMPL_DATA_11_train > 0) = 1;
%         reduced_SAMPL_DATA_11_train(reduced_SAMPL_DATA_11_train < 0) = 2;
        reduced_SAMPL_DATA_12_train = SAMPL_DATA_12_train;
%         reduced_SAMPL_DATA_12_train(reduced_SAMPL_DATA_12_train > 0) = 1;
%         reduced_SAMPL_DATA_12_train(reduced_SAMPL_DATA_12_train < 0) = 2;
        reduced_SAMPL_DATA_11_test = SAMPL_DATA_11_test;
%         reduced_SAMPL_DATA_11_test(reduced_SAMPL_DATA_11_test > 0) = 1;
%         reduced_SAMPL_DATA_11_test(reduced_SAMPL_DATA_11_test < 0) = 2;
        reduced_SAMPL_DATA_12_test = SAMPL_DATA_12_test;
%         reduced_SAMPL_DATA_12_test(reduced_SAMPL_DATA_12_test > 0) = 1;
%         reduced_SAMPL_DATA_12_test(reduced_SAMPL_DATA_12_test < 0) = 2;
        reduced_SAMPL_DATA_21_train = SAMPL_DATA_21_train;
%         reduced_SAMPL_DATA_21_train(reduced_SAMPL_DATA_21_train > 0) = 1;
%         reduced_SAMPL_DATA_21_train(reduced_SAMPL_DATA_21_train < 0) = 2;
        reduced_SAMPL_DATA_22_train = SAMPL_DATA_22_train;
%         reduced_SAMPL_DATA_22_train(reduced_SAMPL_DATA_22_train > 0) = 1;
%         reduced_SAMPL_DATA_22_train(reduced_SAMPL_DATA_22_train < 0) = 2;
        reduced_SAMPL_DATA_21_test = SAMPL_DATA_21_test;
%         reduced_SAMPL_DATA_21_test(reduced_SAMPL_DATA_21_test > 0) = 1;
%         reduced_SAMPL_DATA_21_test(reduced_SAMPL_DATA_21_test < 0) = 2;
        reduced_SAMPL_DATA_22_test = SAMPL_DATA_22_test;
%         reduced_SAMPL_DATA_22_test(reduced_SAMPL_DATA_22_test > 0) = 1;
%         reduced_SAMPL_DATA_22_test(reduced_SAMPL_DATA_22_test < 0) = 2;
        reduced_SAMPL_DATA_31_train = SAMPL_DATA_31_train;
%         reduced_SAMPL_DATA_31_train(reduced_SAMPL_DATA_31_train > 0) = 1;
%         reduced_SAMPL_DATA_31_train(reduced_SAMPL_DATA_31_train < 0) = 2;
        reduced_SAMPL_DATA_32_train = SAMPL_DATA_32_train;
%         reduced_SAMPL_DATA_32_train(reduced_SAMPL_DATA_32_train > 0) = 1;
%         reduced_SAMPL_DATA_32_train(reduced_SAMPL_DATA_32_train < 0) = 2;
        reduced_SAMPL_DATA_31_test = SAMPL_DATA_31_test;
%         reduced_SAMPL_DATA_31_test(reduced_SAMPL_DATA_31_test > 0) = 1;
%         reduced_SAMPL_DATA_31_test(reduced_SAMPL_DATA_31_test < 0) = 2;
        reduced_SAMPL_DATA_32_test = SAMPL_DATA_32_test;
%         reduced_SAMPL_DATA_32_test(reduced_SAMPL_DATA_32_test > 0) = 1;
%         reduced_SAMPL_DATA_32_test(reduced_SAMPL_DATA_32_test < 0) = 2;
        reduced_SAMPL_DATA_41_train = SAMPL_DATA_41_train;
%         reduced_SAMPL_DATA_41_train(reduced_SAMPL_DATA_41_train > 0) = 1;
%         reduced_SAMPL_DATA_41_train(reduced_SAMPL_DATA_41_train < 0) = 2;
        reduced_SAMPL_DATA_42_train = SAMPL_DATA_42_train;
%         reduced_SAMPL_DATA_42_train(reduced_SAMPL_DATA_42_train > 0) = 1;
%         reduced_SAMPL_DATA_42_train(reduced_SAMPL_DATA_42_train < 0) = 2;
        reduced_SAMPL_DATA_41_test = SAMPL_DATA_41_test;
%         reduced_SAMPL_DATA_41_test(reduced_SAMPL_DATA_41_test > 0) = 1;
%         reduced_SAMPL_DATA_41_test(reduced_SAMPL_DATA_41_test < 0) = 2;
        reduced_SAMPL_DATA_42_test = SAMPL_DATA_42_test;
%         reduced_SAMPL_DATA_42_test(reduced_SAMPL_DATA_42_test > 0) = 1;
%         reduced_SAMPL_DATA_42_test(reduced_SAMPL_DATA_42_test < 0) = 2;
        reduced_L_SAMPL_DATA_1_train = L_SAMPL_DATA_01_train - 1;
        reduced_L_SAMPL_DATA_2_train = L_SAMPL_DATA_02_train - 1;
        reduced_L_SAMPL_DATA_1_test = L_SAMPL_DATA_01_test - 1;
        reduced_L_SAMPL_DATA_2_test = L_SAMPL_DATA_02_test - 1;

        % projM1_pos = zeros(channels_v,D);
        % projM3_pos = zeros(channels_v_EEG,D);
        % projM5_pos = zeros(channels_v_GSR,D);
        % projM7_pos = zeros(channels_v_BVP,D);
        % projM9_pos = zeros(channels_v_RES,D);
        % 
        % seed = randCounter;
        % while seed>0
        %     seedVectors = [genRandomHV(D);genRandomHV(D)];
        %     seed = seed - 1;
        % end
        % projMvectors_pos  = containers.Map ('KeyType','double','ValueType','any');
        % for i = 1:total_vectors
        %     vec = seedVectors(1+binary(i,1), :);
        %     for d = 2:length(binary(i,:))
        %         vec = xor(vec, circshift(seedVectors(1+binary(i,d), :),d-1));
        %     end
        %     projMvectors_pos(i) = vec;
        % end
        % 
        % for i=1:1:channels_v_EEG
        %     projM3_pos(i,:) = projMvectors_pos(i);
        % end
        % for i=1:1:channels_v
        %     projM1_pos(i,:) = projM3_pos(i,:);
        % end
        % for i=1:1:channels_v_GSR
        %     projM5_pos(i,:) = projM3_pos(i,:);
        % end
        % for i=1:1:channels_v_BVP
        %     projM7_pos(i,:) = projM3_pos(i,:);
        % end
        % for i=1:1:channels_v_RES
        %     projM9_pos(i,:) = projM3_pos(i,:);
        % end
        % 
        % projM1_neg = zeros(channels_v,D);
        % projM3_neg = zeros(channels_v_EEG,D);
        % projM5_neg = zeros(channels_v_GSR,D);
        % projM7_neg = zeros(channels_v_BVP,D);
        % projM9_neg = zeros(channels_v_RES,D);
        % 
        % seed = randCounter;
        % while seed>0
        %     seedVectors = [genRandomHV(D);genRandomHV(D)];
        %     seed = seed - 1;
        % end
        % projMvectors_neg  = containers.Map ('KeyType','double','ValueType','any');
        % for i = 1:total_vectors
        %     vec = seedVectors(1+binary(i,1), :);
        %     for d = 2:length(binary(i,:))
        %         vec = xor(vec, circshift(seedVectors(1+binary(i,d), :),d-1));
        %     end
        %     projMvectors_neg(i) = vec;
        % end
        % 
        % for i=1:1:channels_v_EEG
        %     projM3_neg(i,:) = projMvectors_neg(i);
        % end
        % for i=1:1:channels_v
        %     projM1_neg(i,:) = projM3_neg(i,:);
        % end
        % for i=1:1:channels_v_GSR
        %     projM5_neg(i,:) = projM3_neg(i,:);
        % end
        % for i=1:1:channels_v_BVP
        %     projM7_neg(i,:) = projM3_neg(i,:);
        % end
        % for i=1:1:channels_v_RES
        %     projM9_neg(i,:) = projM3_neg(i,:);
        % end

%         projM1_neg = projM1;
%         projM1_pos = projM1;
%         projM1_neg(projM1_neg==-1) = 0;
%         projM1_pos(projM1_pos==1) = 0;
%         projM1_pos(projM1==-1) = 1;
%         projM3_neg = projM3;
%         projM3_pos = projM3;
%         projM3_neg(projM3==-1) = 0;
%         projM3_pos(projM3==1) = 0;
%         projM3_pos(projM3==-1) = 1;
%         projM5_neg = projM5;
%         projM5_pos = projM5;
%         projM5_neg(projM5==-1) = 0;
%         projM5_pos(projM5==1) = 0;
%         projM5_pos(projM5==-1) = 1;
%         projM7_neg = projM7;
%         projM7_pos = projM7;
%         projM7_neg(projM7==-1) = 0;
%         projM7_pos(projM7==1) = 0;
%         projM7_pos(projM7==-1) = 1;
%         projM9_neg = projM9;
%         projM9_pos = projM9;
%         projM9_neg(projM9==-1) = 0;
%         projM9_pos(projM9==1) = 0;
%         projM9_pos(projM9==-1) = 1;
                
        %% V
        randCounter
        fprintf ('HDC for V\n');
        if (select == 1)
            [AM, AM1, AM2, AM3, AM4, AM5] = hdctrainproj (classes, reduced_L_SAMPL_DATA_2_train, reduced_SAMPL_DATA_02_train, reduced_SAMPL_DATA_12_train, reduced_SAMPL_DATA_22_train, reduced_SAMPL_DATA_32_train, reduced_SAMPL_DATA_42_train, chAM1, chAM3, chAM5, chAM7, chAM9, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, channels_v, channels_v_EEG, channels_v_GSR, channels_v_BVP, channels_v_RES);
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
        
         [accexc_alltrz, accexc_alltrz_mod1, accexc_alltrz_mod2, accexc_alltrz_mod3, accexc_alltrz_mod4, accexc_alltrz_mod5, accexc_alltrz_majcount, accExcTrnz, accExcTrnz_mod1, accExcTrnz_mod2, accExcTrnz_mod3, accExcTrnz_mod4, accExcTrnz_mod5, accExcTrnz_majcount, accuracy, accuracy_mod1, accuracy_mod2, accuracy_mod3, accuracy_mod4, accuracy_mod5, accuracy_majcount, plv, alv] = hdcpredictproj  (reduced_L_SAMPL_DATA_2_test, reduced_SAMPL_DATA_02_test, reduced_SAMPL_DATA_12_test, reduced_SAMPL_DATA_22_test, reduced_SAMPL_DATA_32_test, reduced_SAMPL_DATA_42_test, AM, AM1, AM2, AM3, AM4, AM5, chAM1, chAM3, chAM5, chAM7, chAM9, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, classes, channels_v, channels_v_EEG, channels_v_GSR, channels_v_BVP, channels_v_RES);

        plv = plv';
        accuracy_majcount
        accuracy
        acc_matrix(((randCounter)*2),subject) = accuracy_majcount;
        acc_matrix_ex(((randCounter)*2),subject) = accExcTrnz_majcount;
        acc_matrix_ex_all(((randCounter)*2),subject) = accexc_alltrz_majcount;
        
        %V matrixes for all versions
        acc_matrix_V(randCounter,subject) = accuracy;
        acc_matrix_V_mod1(randCounter,subject) = accuracy_mod1;
        acc_matrix_V_mod2(randCounter,subject) = accuracy_mod2;
        acc_matrix_V_mod3(randCounter,subject) = accuracy_mod3;
        acc_matrix_V_mod4(randCounter,subject) = accuracy_mod4;
        acc_matrix_V_mod5(randCounter,subject) = accuracy_mod5;
        acc_matrix_V_majcount(randCounter,subject) = accuracy_majcount;
        
        acc_matrix_Aex(randCounter,subject) = accExcTrnz_majcount;
        
        %A matrices excluding transition data for all
        acc_matrix_Vex_all(randCounter,subject) = accexc_alltrz;
        acc_matrix_Vex_all_mod1(randCounter,subject) = accexc_alltrz_mod1;
        acc_matrix_Vex_all_mod2(randCounter,subject) = accexc_alltrz_mod2;
        acc_matrix_Vex_all_mod3(randCounter,subject) = accexc_alltrz_mod3;
        acc_matrix_Vex_all_mod4(randCounter,subject) = accexc_alltrz_mod4;
        acc_matrix_Vex_all_mod5(randCounter,subject) = accexc_alltrz_mod5;
        acc_matrix_Vex_all_majcount(randCounter,subject) = accexc_alltrz_majcount;
        val_right = (plv == alv);

        %% A
        %randCounter
        fprintf ('HDC for A\n');
        if (select == 1)
            [AM, AM1, AM2, AM3, AM4, AM5] = hdctrainproj (classes, reduced_L_SAMPL_DATA_1_train, reduced_SAMPL_DATA_01_train, reduced_SAMPL_DATA_11_train, reduced_SAMPL_DATA_21_train, reduced_SAMPL_DATA_31_train, reduced_SAMPL_DATA_41_train, chAM1, chAM3, chAM5, chAM7, chAM9, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, channels_a, channels_a_EEG, channels_a_GSR, channels_a_BVP, channels_a_RES);
        else
            [numpat_2, hdc_model_2] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_2, chAM8, iMch1, D, N, precision, channels_a,projM1_pos,projM1_neg, classes); 
            [numpat_4, hdc_model_4] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_4, chAM8, iMch3, D, N, precision, channels_a_ECG,projM3_pos,projM3_neg, classes); 
            [numpat_6, hdc_model_6] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_6, chAM8, iMch5, D, N, precision, channels_a_EEG,projM5_pos,projM5_neg, classes); 
        end

        %bundle all the sensors (this is the fusion point for late fusion)
        if (select ~= 1)
            %class 1
            hdc_model_2(0)=mode([hdc_model_2(0); hdc_model_4(0); hdc_model_6(0)]);
            %class 2
            hdc_model_2(1)=mode([hdc_model_2(1); hdc_model_4(1); hdc_model_6(1)]);
        end

        [accexc_alltrz, accexc_alltrz_mod1, accexc_alltrz_mod2, accexc_alltrz_mod3, accexc_alltrz_mod4, accexc_alltrz_mod5, accexc_alltrz_majcount, accExcTrnz, accExcTrnz_mod1, accExcTrnz_mod2, accExcTrnz_mod3, accExcTrnz_mod4, accExcTrnz_mod5, accExcTrnz_majcount, accuracy, accuracy_mod1, accuracy_mod2, accuracy_mod3, accuracy_mod4, accuracy_mod5, accuracy_majcount, pla, ala] = hdcpredictproj  (reduced_L_SAMPL_DATA_1_test, reduced_SAMPL_DATA_01_test, reduced_SAMPL_DATA_11_test, reduced_SAMPL_DATA_21_test, reduced_SAMPL_DATA_31_test, reduced_SAMPL_DATA_41_test, AM, AM1, AM2, AM3, AM4, AM5, chAM1, chAM3, chAM5, chAM7, chAM9, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, classes, channels_a, channels_a_EEG, channels_a_GSR, channels_a_BVP, channels_a_RES);

        pla = pla';
        accuracy_majcount
        accuracy
        %big matrixes have majority count accuracies
        acc_matrix(((randCounter)*2-1),subject) = accuracy_majcount;
        acc_matrix_ex(((randCounter)*2-1),subject) = accExcTrnz_majcount;
        acc_matrix_ex_all(((randCounter)*2-1),subject) = accexc_alltrz_majcount;
        
        %A matrixes for all versions
        acc_matrix_A(randCounter,subject) = accuracy;
        acc_matrix_A_mod1(randCounter,subject) = accuracy_mod1;
        acc_matrix_A_mod2(randCounter,subject) = accuracy_mod2;
        acc_matrix_A_mod3(randCounter,subject) = accuracy_mod3;
        acc_matrix_A_mod4(randCounter,subject) = accuracy_mod4;
        acc_matrix_A_mod5(randCounter,subject) = accuracy_mod5;
        acc_matrix_A_majcount(randCounter,subject) = accuracy_majcount;
        
        acc_matrix_Aex(randCounter,subject) = accExcTrnz_majcount;
        
        %A matrices excluding transition data for all
        acc_matrix_Aex_all(randCounter,subject) = accexc_alltrz;
        acc_matrix_Aex_all_mod1(randCounter,subject) = accexc_alltrz_mod1;
        acc_matrix_Aex_all_mod2(randCounter,subject) = accexc_alltrz_mod2;
        acc_matrix_Aex_all_mod3(randCounter,subject) = accexc_alltrz_mod3;
        acc_matrix_Aex_all_mod4(randCounter,subject) = accexc_alltrz_mod4;
        acc_matrix_Aex_all_mod5(randCounter,subject) = accexc_alltrz_mod5;
        acc_matrix_Aex_all_majcount(randCounter,subject) = accexc_alltrz_majcount;
        
        
        ar_right = (pla == ala);

        % iMfull = [];
        % for i = 1:1:iMch9.Count
        %     iMfull = [iMfull iMch9(i)]; %#ok<AGROW>
        % end
        % 
        % projM_pos_full = [];
        % %projM_pos_temp = [projM1_pos; projM3_pos; projM5_pos];
        % x = size(projM9_pos);
        % dim = x(1);
        % for i = 1:1:dim
        %     projM_pos_full = [projM_pos_full projM9_pos(i,:)]; %#ok<AGROW>
        % end
        % 
        % projM_neg_full = [];
        % %projM_neg_temp = [projM1_neg; projM3_neg; projM5_neg];
        % x = size(projM9_neg);
        % dim = x(1);
        % for i = 1:1:dim
        %     projM_neg_full = [projM_neg_full projM9_neg(i,:)]; %#ok<AGROW>
        % end

    end
    randCounter=randCounter-1;
end
%normal accuracy
% acc_av_A_matrix = mean(acc_matrix_A);
% acc_av_V_matrix = mean(acc_matrix_V);
% acc_OverallAav = mean(acc_av_A_matrix)
% acc_OverallVav = mean(acc_av_V_matrix)
% acc_max_A_matrix = max(acc_matrix_A);
% acc_max_V_matrix = max(acc_matrix_V);
% acc_OverallAmax = mean(acc_max_A_matrix);
% acc_OverallVmax = mean(acc_max_V_matrix);

%ex accuracy
% acc_av_ex_A_matrix = mean(acc_matrix_Aex);
% acc_av_ex_V_matrix = mean(acc_matrix_Vex);
% acc_OverallAav_ex = mean(acc_av_ex_A_matrix)
% acc_OverallVav_ex = mean(acc_av_ex_V_matrix)
% acc_max_ex_A_matrix = max(acc_matrix_Aex);
% acc_max_ex_V_matrix = max(acc_matrix_Vex);
% acc_OverallAmax_ex = mean(acc_max_ex_A_matrix);
% acc_OverallVmax_ex = mean(acc_max_ex_V_matrix);

%excluding transition data from all counts
% acc_av_ex_all_A_matrix = mean(acc_matrix_Aex_all);
% acc_av_ex_all_V_matrix = mean(acc_matrix_Vex_all);
% acc_OverallAav_ex_all = mean(acc_av_ex_all_A_matrix)
% acc_OverallVav_ex_all = mean(acc_av_ex_all_V_matrix)
% acc_max_ex_all_A_matrix = max(acc_matrix_Aex_all);
% acc_max_ex_all_V_matrix = max(acc_matrix_Vex_all);
% acc_OverallAmax_ex_all = mean(acc_max_ex_all_A_matrix);
% acc_OverallVmax_ex_all = mean(acc_max_ex_all_V_matrix);

%% horizontal metrics (average across the 32 subjects first)

%normal accuracy
mean_A = mean(acc_matrix_A,2);
mean_V = mean(acc_matrix_V,2);
mean_AV = (mean_A + mean_V)./2;
[~,max_index] = max(mean_AV);
%% key metrics
acc_av_max_A = mean_A(max_index);
acc_av_max_V = mean_V(max_index);
acc_max_V = max(mean_V);
acc_max_A = max(mean_A);
acc_av_av_A = mean(mean_A);
acc_av_av_V = mean(mean_V);
aaaaccuracy_sprdsht = zeros(3,14);
aaaaccuracy_sprdsht(1,1) = acc_av_av_A;
aaaaccuracy_sprdsht(1,2) = acc_av_av_V;
aaaaccuracy_sprdsht(2,1) = acc_max_A;
aaaaccuracy_sprdsht(2,2) = acc_max_V;
aaaaccuracy_sprdsht(3,1) = acc_av_max_A;
aaaaccuracy_sprdsht(3,2) = acc_av_max_V;

%majority count
mean_A = mean(acc_matrix_A_majcount,2);
mean_V = mean(acc_matrix_V_majcount,2);
mean_AV = (mean_A + mean_V)./2;
[~,max_index] = max(mean_AV);
%% key metrics
acc_av_max_A = mean_A(max_index);
acc_av_max_V = mean_V(max_index);
acc_max_V = max(mean_V);
acc_max_A = max(mean_A);
acc_av_av_A = mean(mean_A);
acc_av_av_V = mean(mean_V);
aaaaccuracy_sprdsht(1,3) = acc_av_av_A;
aaaaccuracy_sprdsht(1,4) = acc_av_av_V;
aaaaccuracy_sprdsht(2,3) = acc_max_A;
aaaaccuracy_sprdsht(2,4) = acc_max_V;
aaaaccuracy_sprdsht(3,3) = acc_av_max_A;
aaaaccuracy_sprdsht(3,4) = acc_av_max_V;

%modality1
mean_A = mean(acc_matrix_A_mod1,2);
mean_V = mean(acc_matrix_V_mod1,2);
mean_AV = (mean_A + mean_V)./2;
[~,max_index] = max(mean_AV);
%% key metrics
acc_av_max_A = mean_A(max_index);
acc_av_max_V = mean_V(max_index);
acc_max_V = max(mean_V);
acc_max_A = max(mean_A);
acc_av_av_A = mean(mean_A);
acc_av_av_V = mean(mean_V);
aaaaccuracy_sprdsht(1,5) = acc_av_av_A;
aaaaccuracy_sprdsht(1,6) = acc_av_av_V;
aaaaccuracy_sprdsht(2,5) = acc_max_A;
aaaaccuracy_sprdsht(2,6) = acc_max_V;
aaaaccuracy_sprdsht(3,5) = acc_av_max_A;
aaaaccuracy_sprdsht(3,6) = acc_av_max_V;

%modality2
mean_A = mean(acc_matrix_A_mod2,2);
mean_V = mean(acc_matrix_V_mod2,2);
mean_AV = (mean_A + mean_V)./2;
[~,max_index] = max(mean_AV);
%% key metrics
acc_av_max_A = mean_A(max_index);
acc_av_max_V = mean_V(max_index);
acc_max_V = max(mean_V);
acc_max_A = max(mean_A);
acc_av_av_A = mean(mean_A);
acc_av_av_V = mean(mean_V);
aaaaccuracy_sprdsht(1,7) = acc_av_av_A;
aaaaccuracy_sprdsht(1,8) = acc_av_av_V;
aaaaccuracy_sprdsht(2,7) = acc_max_A;
aaaaccuracy_sprdsht(2,8) = acc_max_V;
aaaaccuracy_sprdsht(3,7) = acc_av_max_A;
aaaaccuracy_sprdsht(3,8) = acc_av_max_V;

%modality3
mean_A = mean(acc_matrix_A_mod3,2);
mean_V = mean(acc_matrix_V_mod3,2);
mean_AV = (mean_A + mean_V)./2;
[~,max_index] = max(mean_AV);
%% key metrics
acc_av_max_A = mean_A(max_index);
acc_av_max_V = mean_V(max_index);
acc_max_V = max(mean_V);
acc_max_A = max(mean_A);
acc_av_av_A = mean(mean_A);
acc_av_av_V = mean(mean_V);
aaaaccuracy_sprdsht(1,9) = acc_av_av_A;
aaaaccuracy_sprdsht(1,10) = acc_av_av_V;
aaaaccuracy_sprdsht(2,9) = acc_max_A;
aaaaccuracy_sprdsht(2,10) = acc_max_V;
aaaaccuracy_sprdsht(3,9) = acc_av_max_A;
aaaaccuracy_sprdsht(3,10) = acc_av_max_V;

%modality4
mean_A = mean(acc_matrix_A_mod4,2);
mean_V = mean(acc_matrix_V_mod4,2);
mean_AV = (mean_A + mean_V)./2;
[~,max_index] = max(mean_AV);
%% key metrics
acc_av_max_A = mean_A(max_index);
acc_av_max_V = mean_V(max_index);
acc_max_V = max(mean_V);
acc_max_A = max(mean_A);
acc_av_av_A = mean(mean_A);
acc_av_av_V = mean(mean_V);
aaaaccuracy_sprdsht(1,11) = acc_av_av_A;
aaaaccuracy_sprdsht(1,12) = acc_av_av_V;
aaaaccuracy_sprdsht(2,11) = acc_max_A;
aaaaccuracy_sprdsht(2,12) = acc_max_V;
aaaaccuracy_sprdsht(3,11) = acc_av_max_A;
aaaaccuracy_sprdsht(3,12) = acc_av_max_V;

%modality5
mean_A = mean(acc_matrix_A_mod5,2);
mean_V = mean(acc_matrix_V_mod5,2);
mean_AV = (mean_A + mean_V)./2;
[~,max_index] = max(mean_AV);
%% key metrics
acc_av_max_A = mean_A(max_index);
acc_av_max_V = mean_V(max_index);
acc_max_V = max(mean_V);
acc_max_A = max(mean_A);
acc_av_av_A = mean(mean_A);
acc_av_av_V = mean(mean_V);
aaaaccuracy_sprdsht(1,13) = acc_av_av_A;
aaaaccuracy_sprdsht(1,14) = acc_av_av_V;
aaaaccuracy_sprdsht(2,13) = acc_max_A;
aaaaccuracy_sprdsht(2,14) = acc_max_V;
aaaaccuracy_sprdsht(3,13) = acc_av_max_A;
aaaaccuracy_sprdsht(3,14) = acc_av_max_V;

% ex accuracy
% mean_A_ex = mean(acc_matrix_Aex,2);
% mean_V_ex = mean(acc_matrix_Vex,2);
% mean_AV_ex = (mean_A_ex + mean_V_ex)./2;
% [~,max_index_ex] = max(mean_AV_ex);
% acc_av_max_A_ex = mean_A_ex(max_index_ex);
% acc_av_max_V_ex = mean_V_ex(max_index_ex);
% acc_max_V_ex = max(mean_V_ex);
% acc_max_A_ex = max(mean_A_ex);

%excluding transition data from all counts
%normal
mean_A_ex_all = mean(acc_matrix_Aex_all,2);
mean_V_ex_all = mean(acc_matrix_Vex_all,2);
mean_AV_ex_all = (mean_A_ex_all + mean_V_ex_all)./2;
[~,max_index_ex_all] = max(mean_AV_ex_all);
%% plot this one
aaacc_av_max_A_ex_all = mean_A_ex_all(max_index_ex_all);
aaacc_av_max_V_ex_all = mean_V_ex_all(max_index_ex_all);
%%
aaacc_max_V_ex_all = max(mean_V_ex_all);
aaacc_max_A_ex_all = max(mean_A_ex_all);
%% plot this one
aacc_av_av_A_ex_all = mean(mean_A_ex_all);
aacc_av_av_V_ex_all = mean(mean_V_ex_all);

aaaaccuracy_sprdsht_ex_all = zeros(3,2);
aaaaccuracy_sprdsht_ex_all(1,1) = aacc_av_av_A_ex_all;
aaaaccuracy_sprdsht_ex_all(1,2) = aacc_av_av_V_ex_all;
aaaaccuracy_sprdsht_ex_all(2,1) = aaacc_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(2,2) = aaacc_max_V_ex_all;
aaaaccuracy_sprdsht_ex_all(3,1) = aaacc_av_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(3,2) = aaacc_av_max_V_ex_all;

%majority count
mean_A_ex_all = mean(acc_matrix_Aex_all_majcount,2);
mean_V_ex_all = mean(acc_matrix_Vex_all_majcount,2);
mean_AV_ex_all = (mean_A_ex_all + mean_V_ex_all)./2;
[~,max_index_ex_all] = max(mean_AV_ex_all);
aaacc_av_max_A_ex_all = mean_A_ex_all(max_index_ex_all);
aaacc_av_max_V_ex_all = mean_V_ex_all(max_index_ex_all);
aaacc_max_V_ex_all = max(mean_V_ex_all);
aaacc_max_A_ex_all = max(mean_A_ex_all);
aacc_av_av_A_ex_all = mean(mean_A_ex_all);
aacc_av_av_V_ex_all = mean(mean_V_ex_all);
aaaaccuracy_sprdsht_ex_all(1,3) = aacc_av_av_A_ex_all;
aaaaccuracy_sprdsht_ex_all(1,4) = aacc_av_av_V_ex_all;
aaaaccuracy_sprdsht_ex_all(2,3) = aaacc_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(2,4) = aaacc_max_V_ex_all;
aaaaccuracy_sprdsht_ex_all(3,3) = aaacc_av_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(3,4) = aaacc_av_max_V_ex_all;

%modality1 
mean_A_ex_all = mean(acc_matrix_Aex_all_mod1,2);
mean_V_ex_all = mean(acc_matrix_Vex_all_mod1,2);
mean_AV_ex_all = (mean_A_ex_all + mean_V_ex_all)./2;
[~,max_index_ex_all] = max(mean_AV_ex_all);
aaacc_av_max_A_ex_all = mean_A_ex_all(max_index_ex_all);
aaacc_av_max_V_ex_all = mean_V_ex_all(max_index_ex_all);
aaacc_max_V_ex_all = max(mean_V_ex_all);
aaacc_max_A_ex_all = max(mean_A_ex_all);
aacc_av_av_A_ex_all = mean(mean_A_ex_all);
aacc_av_av_V_ex_all = mean(mean_V_ex_all);
aaaaccuracy_sprdsht_ex_all(1,5) = aacc_av_av_A_ex_all;
aaaaccuracy_sprdsht_ex_all(1,6) = aacc_av_av_V_ex_all;
aaaaccuracy_sprdsht_ex_all(2,5) = aaacc_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(2,6) = aaacc_max_V_ex_all;
aaaaccuracy_sprdsht_ex_all(3,5) = aaacc_av_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(3,6) = aaacc_av_max_V_ex_all;

%modality2 
mean_A_ex_all = mean(acc_matrix_Aex_all_mod2,2);
mean_V_ex_all = mean(acc_matrix_Vex_all_mod2,2);
mean_AV_ex_all = (mean_A_ex_all + mean_V_ex_all)./2;
[~,max_index_ex_all] = max(mean_AV_ex_all);
aaacc_av_max_A_ex_all = mean_A_ex_all(max_index_ex_all);
aaacc_av_max_V_ex_all = mean_V_ex_all(max_index_ex_all);
aaacc_max_V_ex_all = max(mean_V_ex_all);
aaacc_max_A_ex_all = max(mean_A_ex_all);
aacc_av_av_A_ex_all = mean(mean_A_ex_all);
aacc_av_av_V_ex_all = mean(mean_V_ex_all);
aaaaccuracy_sprdsht_ex_all(1,7) = aacc_av_av_A_ex_all;
aaaaccuracy_sprdsht_ex_all(1,8) = aacc_av_av_V_ex_all;
aaaaccuracy_sprdsht_ex_all(2,7) = aaacc_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(2,8) = aaacc_max_V_ex_all;
aaaaccuracy_sprdsht_ex_all(3,7) = aaacc_av_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(3,8) = aaacc_av_max_V_ex_all;

%modality3
mean_A_ex_all = mean(acc_matrix_Aex_all_mod3,2);
mean_V_ex_all = mean(acc_matrix_Vex_all_mod3,2);
mean_AV_ex_all = (mean_A_ex_all + mean_V_ex_all)./2;
[~,max_index_ex_all] = max(mean_AV_ex_all);
aaacc_av_max_A_ex_all = mean_A_ex_all(max_index_ex_all);
aaacc_av_max_V_ex_all = mean_V_ex_all(max_index_ex_all);
aaacc_max_V_ex_all = max(mean_V_ex_all);
aaacc_max_A_ex_all = max(mean_A_ex_all);
aacc_av_av_A_ex_all = mean(mean_A_ex_all);
aacc_av_av_V_ex_all = mean(mean_V_ex_all);
aaaaccuracy_sprdsht_ex_all(1,9) = aacc_av_av_A_ex_all;
aaaaccuracy_sprdsht_ex_all(1,10) = aacc_av_av_V_ex_all;
aaaaccuracy_sprdsht_ex_all(2,9) = aaacc_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(2,10) = aaacc_max_V_ex_all;
aaaaccuracy_sprdsht_ex_all(3,9) = aaacc_av_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(3,10) = aaacc_av_max_V_ex_all;

%modality4
mean_A_ex_all = mean(acc_matrix_Aex_all_mod4,2);
mean_V_ex_all = mean(acc_matrix_Vex_all_mod4,2);
mean_AV_ex_all = (mean_A_ex_all + mean_V_ex_all)./2;
[~,max_index_ex_all] = max(mean_AV_ex_all);
aaacc_av_max_A_ex_all = mean_A_ex_all(max_index_ex_all);
aaacc_av_max_V_ex_all = mean_V_ex_all(max_index_ex_all);
aaacc_max_V_ex_all = max(mean_V_ex_all);
aaacc_max_A_ex_all = max(mean_A_ex_all);
aacc_av_av_A_ex_all = mean(mean_A_ex_all);
aacc_av_av_V_ex_all = mean(mean_V_ex_all);
aaaaccuracy_sprdsht_ex_all(1,11) = aacc_av_av_A_ex_all;
aaaaccuracy_sprdsht_ex_all(1,12) = aacc_av_av_V_ex_all;
aaaaccuracy_sprdsht_ex_all(2,11) = aaacc_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(2,12) = aaacc_max_V_ex_all;
aaaaccuracy_sprdsht_ex_all(3,11) = aaacc_av_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(3,12) = aaacc_av_max_V_ex_all;

%modality5
mean_A_ex_all = mean(acc_matrix_Aex_all_mod5,2);
mean_V_ex_all = mean(acc_matrix_Vex_all_mod5,2);
mean_AV_ex_all = (mean_A_ex_all + mean_V_ex_all)./2;
[~,max_index_ex_all] = max(mean_AV_ex_all);
aaacc_av_max_A_ex_all = mean_A_ex_all(max_index_ex_all);
aaacc_av_max_V_ex_all = mean_V_ex_all(max_index_ex_all);
aaacc_max_V_ex_all = max(mean_V_ex_all);
aaacc_max_A_ex_all = max(mean_A_ex_all);
aacc_av_av_A_ex_all = mean(mean_A_ex_all);
aacc_av_av_V_ex_all = mean(mean_V_ex_all);
aaaaccuracy_sprdsht_ex_all(1,13) = aacc_av_av_A_ex_all;
aaaaccuracy_sprdsht_ex_all(1,14) = aacc_av_av_V_ex_all;
aaaaccuracy_sprdsht_ex_all(2,13) = aaacc_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(2,14) = aaacc_max_V_ex_all;
aaaaccuracy_sprdsht_ex_all(3,13) = aaacc_av_max_A_ex_all;
aaaaccuracy_sprdsht_ex_all(3,14) = aaacc_av_max_V_ex_all;

%%also put into spreadsheet
aaaaccuracy_matrix_data = acc_matrix;
aaaaccuracy_matrix_data_ex_all = acc_matrix_ex_all;