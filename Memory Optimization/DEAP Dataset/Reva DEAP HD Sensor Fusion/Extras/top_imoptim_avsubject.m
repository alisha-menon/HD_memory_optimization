clear;
randCounter= 2; %per subject
full_count = randCounter;
learningrate=0.5; % percentage of the dataset used to train the algorithm
downSampRate = 1;
ngram = 3; % for temporal encode
subjects = 3;
%====Features and Label===
load('DEAP_data.mat')
acc_matrix = zeros(randCounter*2,subjects);
acc_matrix_A = zeros(randCounter,subjects);
acc_matrix_V = zeros(randCounter,subjects);
acc_matrix_Vex = zeros(randCounter,subjects);
acc_matrix_Aex = zeros(randCounter,subjects);
acc_matrix_Vex_all = zeros(randCounter,subjects);
acc_matrix_Aex_all = zeros(randCounter,subjects);

for subject = 1:1:subjects
    subject
    features=inputs((subject-1)*40+1:subject*40,:);
    f_label_a_binary=features(:,239);
    f_label_v_binary=features(:,240);
    % f_label_d_binary=features(:,241);
    % f_label_l_binary=features(:,242);

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
    % f_label_d_binary(f_label_d_binary < m) = 1;
    % f_label_d_binary(f_label_d_binary >= m) = 2;
    % f_label_l_binary(f_label_l_binary < m) = 1;
    % f_label_l_binary(f_label_l_binary >= m) = 2;

    %met_A_accuracy = zeros(1,12);
    %met_V_accuracy = zeros(1,12);
    % met_D_accuracy = zeros(1,12);
    % met_L_accuracy = zeros(1,12);

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
     features(:,i)=features(:,i)+0.35;
    end

    features_EMG=features(:,1:10);
    features_EEG=features(:,11:202);
    features_GSR=features(:,203:209);
    features_BVP=features(:,210:226);
    features_RES=features(:,227:238);


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
        HD_functions_mod_reduced;     % load HD functions
    else 
        HD_functions_multiplex;
    end

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
    acc_hamclass_A = zeros(full_count,1);
    acc_hamclass_V = zeros(full_count,1);
    prediction_values_V = zeros(size(features,1),full_count);
    prediction_values_A = zeros(size(features,1),full_count);
    actual_values_V = zeros(size(features,1),full_count);
    actual_values_A = zeros(size(features,1),full_count);

    while (randCounter>0)
        rng('shuffle');
        for j=1:length(D_full)
            learningFrac = learningrate(1); 
            %learningFrac;
            D=D_full(j);
            D
            classes = 2; % level of classes
            precision = 20; %no use
            maxL = 2; % for IM gen
            channels_v_EXG=channels_v+channels_v_EEG+channels_v_GSR+channels_v_BVP+channels_v_RES;
            channels_a_EXG=channels_a+channels_a_EEG+channels_a_GSR+channels_a_BVP+channels_a_RES;
            channels_d_EXG=channels_d+channels_d_EEG+channels_d_GSR+channels_d_BVP+channels_d_RES;
            channels_l_EXG=channels_l+channels_l_EEG+channels_l_GSR+channels_l_BVP+channels_l_RES;


            [chAM1, iMch1] = initItemMemories (D, maxL, channels_v);
            [chAM3, iMch3] = initItemMemories (D, maxL, channels_v_EEG);
            [chAM5, iMch5] = initItemMemories (D, maxL, channels_v_GSR);
            [chAM7, iMch7] = initItemMemories (D, maxL, channels_v_BVP);
            [chAM9, iMch9] = initItemMemories (D, maxL, channels_v_RES);

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
            [chAM8, iMch8] = initItemMemories (D, maxL, channels_a_EXG);

            %downsample the dataset using the value contained in the variable "downSampRate"
            %returns downsampled data which skips every 8 of the original dataset
            LABEL_1_v=f_label_v_binary;
            LABEL_1_a=f_label_a_binary;
            % LABEL_1_d=f_label_d_binary;
            % LABEL_1_l=f_label_l_binary;

            [TS_COMPLETE_01, L_TS_COMPLETE_01] = downSampling (COMPLETE_1_v, LABEL_1_v, downSampRate); %emg
            [TS_COMPLETE_02, L_TS_COMPLETE_02] = downSampling (COMPLETE_1_a, LABEL_1_a, downSampRate);
            % [TS_COMPLETE_03, L_TS_COMPLETE_03] = downSampling (COMPLETE_1_d, LABEL_1_d, downSampRate);
            % [TS_COMPLETE_04, L_TS_COMPLETE_04] = downSampling (COMPLETE_1_l, LABEL_1_l, downSampRate);

            [TS_COMPLETE_11, L_TS_COMPLETE_11] = downSampling (COMPLETE_1_v_EEG, LABEL_1_v, downSampRate);
            [TS_COMPLETE_12, L_TS_COMPLETE_12] = downSampling (COMPLETE_1_a_EEG, LABEL_1_a, downSampRate);
            % [TS_COMPLETE_13, L_TS_COMPLETE_13] = downSampling (COMPLETE_1_d_EEG, LABEL_1_d, downSampRate);
            % [TS_COMPLETE_14, L_TS_COMPLETE_14] = downSampling (COMPLETE_1_l_EEG, LABEL_1_l, downSampRate);

            [TS_COMPLETE_21, L_TS_COMPLETE_21] = downSampling (COMPLETE_1_v_GSR, LABEL_1_v, downSampRate);
            [TS_COMPLETE_22, L_TS_COMPLETE_22] = downSampling (COMPLETE_1_a_GSR, LABEL_1_a, downSampRate);
            % [TS_COMPLETE_23, L_TS_COMPLETE_23] = downSampling (COMPLETE_1_d_GSR, LABEL_1_d, downSampRate);
            % [TS_COMPLETE_24, L_TS_COMPLETE_24] = downSampling (COMPLETE_1_l_GSR, LABEL_1_l, downSampRate);

            [TS_COMPLETE_31, L_TS_COMPLETE_31] = downSampling (COMPLETE_1_v_BVP, LABEL_1_v, downSampRate);
            [TS_COMPLETE_32, L_TS_COMPLETE_32] = downSampling (COMPLETE_1_a_BVP, LABEL_1_a, downSampRate);
            % [TS_COMPLETE_33, L_TS_COMPLETE_33] = downSampling (COMPLETE_1_d_BVP, LABEL_1_d, downSampRate);
            % [TS_COMPLETE_34, L_TS_COMPLETE_34] = downSampling (COMPLETE_1_l_BVP, LABEL_1_l, downSampRate);

            [TS_COMPLETE_41, L_TS_COMPLETE_41] = downSampling (COMPLETE_1_v_RES, LABEL_1_v, downSampRate);
            [TS_COMPLETE_42, L_TS_COMPLETE_42] = downSampling (COMPLETE_1_a_RES, LABEL_1_a, downSampRate);
            % [TS_COMPLETE_43, L_TS_COMPLETE_43] = downSampling (COMPLETE_1_d_RES, LABEL_1_d, downSampRate);
            % [TS_COMPLETE_44, L_TS_COMPLETE_44] = downSampling (COMPLETE_1_l_RES, LABEL_1_l, downSampRate);


            reduced_TS_COMPLETE_01 = TS_COMPLETE_01;
            reduced_TS_COMPLETE_01(reduced_TS_COMPLETE_01 > 0) = 1;
            reduced_TS_COMPLETE_01(reduced_TS_COMPLETE_01 < 0) = 2;
            reduced_TS_COMPLETE_02 = TS_COMPLETE_02;
            reduced_TS_COMPLETE_02(reduced_TS_COMPLETE_02 > 0) = 1;
            reduced_TS_COMPLETE_02(reduced_TS_COMPLETE_02 < 0) = 2;
            % reduced_TS_COMPLETE_03 = TS_COMPLETE_03;
            % reduced_TS_COMPLETE_03(reduced_TS_COMPLETE_03 > 0) = 1;
            % reduced_TS_COMPLETE_03(reduced_TS_COMPLETE_03 < 0) = 2;
            % reduced_TS_COMPLETE_04 = TS_COMPLETE_04;
            % reduced_TS_COMPLETE_04(reduced_TS_COMPLETE_04 > 0) = 1;
            % reduced_TS_COMPLETE_04(reduced_TS_COMPLETE_04 < 0) = 2;
            reduced_TS_COMPLETE_11 = TS_COMPLETE_11;
            reduced_TS_COMPLETE_11(reduced_TS_COMPLETE_11 > 0) = 1;
            reduced_TS_COMPLETE_11(reduced_TS_COMPLETE_11 < 0) = 2;
            reduced_TS_COMPLETE_12 = TS_COMPLETE_12;
            reduced_TS_COMPLETE_12(reduced_TS_COMPLETE_12 > 0) = 1;
            reduced_TS_COMPLETE_12(reduced_TS_COMPLETE_12 < 0) = 2;
            % reduced_TS_COMPLETE_13 = TS_COMPLETE_13;
            % reduced_TS_COMPLETE_13(reduced_TS_COMPLETE_13 > 0) = 1;
            % reduced_TS_COMPLETE_13(reduced_TS_COMPLETE_13 < 0) = 2;
            % reduced_TS_COMPLETE_14 = TS_COMPLETE_14;
            % reduced_TS_COMPLETE_14(reduced_TS_COMPLETE_14 > 0) = 1;
            % reduced_TS_COMPLETE_14(reduced_TS_COMPLETE_14 < 0) = 2;
            reduced_TS_COMPLETE_21 = TS_COMPLETE_21;
            reduced_TS_COMPLETE_21(reduced_TS_COMPLETE_21 > 0) = 1;
            reduced_TS_COMPLETE_21(reduced_TS_COMPLETE_21 < 0) = 2;
            reduced_TS_COMPLETE_22 = TS_COMPLETE_22;
            reduced_TS_COMPLETE_22(reduced_TS_COMPLETE_22 > 0) = 1;
            reduced_TS_COMPLETE_22(reduced_TS_COMPLETE_22 < 0) = 2;
            % reduced_TS_COMPLETE_23 = TS_COMPLETE_23;
            % reduced_TS_COMPLETE_23(reduced_TS_COMPLETE_23 > 0) = 1;
            % reduced_TS_COMPLETE_23(reduced_TS_COMPLETE_23 < 0) = 2;
            % reduced_TS_COMPLETE_24 = TS_COMPLETE_24;
            % reduced_TS_COMPLETE_24(reduced_TS_COMPLETE_24 > 0) = 1;
            % reduced_TS_COMPLETE_24(reduced_TS_COMPLETE_24 < 0) = 2;
            reduced_TS_COMPLETE_31 = TS_COMPLETE_31;
            reduced_TS_COMPLETE_31(reduced_TS_COMPLETE_31 > 0) = 1;
            reduced_TS_COMPLETE_31(reduced_TS_COMPLETE_31 < 0) = 2;
            reduced_TS_COMPLETE_32 = TS_COMPLETE_32;
            reduced_TS_COMPLETE_32(reduced_TS_COMPLETE_32 > 0) = 1;
            reduced_TS_COMPLETE_32(reduced_TS_COMPLETE_32 < 0) = 2;
            % reduced_TS_COMPLETE_33 = TS_COMPLETE_33;
            % reduced_TS_COMPLETE_33(reduced_TS_COMPLETE_33 > 0) = 1;
            % reduced_TS_COMPLETE_33(reduced_TS_COMPLETE_33 < 0) = 2;
            % reduced_TS_COMPLETE_34 = TS_COMPLETE_34;
            % reduced_TS_COMPLETE_34(reduced_TS_COMPLETE_34 > 0) = 1;
            % reduced_TS_COMPLETE_34(reduced_TS_COMPLETE_34 < 0) = 2;
            reduced_TS_COMPLETE_41 = TS_COMPLETE_41;
            reduced_TS_COMPLETE_41(reduced_TS_COMPLETE_41 > 0) = 1;
            reduced_TS_COMPLETE_41(reduced_TS_COMPLETE_41 < 0) = 2;
            reduced_TS_COMPLETE_42 = TS_COMPLETE_42;
            reduced_TS_COMPLETE_42(reduced_TS_COMPLETE_42 > 0) = 1;
            reduced_TS_COMPLETE_42(reduced_TS_COMPLETE_42 < 0) = 2;
            % reduced_TS_COMPLETE_43 = TS_COMPLETE_43;
            % reduced_TS_COMPLETE_43(reduced_TS_COMPLETE_43 > 0) = 1;
            % reduced_TS_COMPLETE_43(reduced_TS_COMPLETE_43 < 0) = 2;
            % reduced_TS_COMPLETE_44 = TS_COMPLETE_44;
            % reduced_TS_COMPLETE_44(reduced_TS_COMPLETE_44 > 0) = 1;
            % reduced_TS_COMPLETE_44(reduced_TS_COMPLETE_44 < 0) = 2;


            reduced_L_TS_COMPLETE_1 = L_TS_COMPLETE_01;
            reduced_L_TS_COMPLETE_1(reduced_L_TS_COMPLETE_1 == 1) = 0;
            reduced_L_TS_COMPLETE_1(reduced_L_TS_COMPLETE_1 == 2) = 1;
            reduced_L_TS_COMPLETE_2 = L_TS_COMPLETE_02;
            reduced_L_TS_COMPLETE_2(reduced_L_TS_COMPLETE_2 == 1) = 0;
            reduced_L_TS_COMPLETE_2(reduced_L_TS_COMPLETE_2 == 2) = 1;
            % reduced_L_TS_COMPLETE_3 = L_TS_COMPLETE_03;
            % reduced_L_TS_COMPLETE_3(reduced_L_TS_COMPLETE_3 == 1) = 0;
            % reduced_L_TS_COMPLETE_3(reduced_L_TS_COMPLETE_3 == 2) = 1;
            % reduced_L_TS_COMPLETE_4 = L_TS_COMPLETE_04;
            % reduced_L_TS_COMPLETE_4(reduced_L_TS_COMPLETE_4 == 1) = 0;
            % reduced_L_TS_COMPLETE_4(reduced_L_TS_COMPLETE_4 == 2) = 1;

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

            % dominance_count_class_change = 0;
            % for i = 1:1:length(LABEL_1_d)-1
            %     if LABEL_1_d(i) ~= LABEL_1_d(i+1)
            %         dominance_count_class_change = dominance_count_class_change+1;
            %     end
            % end
            % 
            % liking_count_class_change = 0;
            % for i = 1:1:length(LABEL_1_l)-1
            %     if LABEL_1_l(i) ~= LABEL_1_l(i+1)
            %         liking_count_class_change = liking_count_class_change+1;
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
            % [L_SAMPL_DATA_01, SAMPL_DATA_01] = genTrainData (TS_COMPLETE_01, L_TS_COMPLETE_01, learningFrac, 'inorder');
            % [L_SAMPL_DATA_02, SAMPL_DATA_02] = genTrainData (TS_COMPLETE_02, L_TS_COMPLETE_02, learningFrac, 'inorder');
            % [L_SAMPL_DATA_11, SAMPL_DATA_11] = genTrainData (TS_COMPLETE_11, L_TS_COMPLETE_11, learningFrac, 'inorder');
            % [L_SAMPL_DATA_12, SAMPL_DATA_12] = genTrainData (TS_COMPLETE_12, L_TS_COMPLETE_12, learningFrac, 'inorder');
            % [L_SAMPL_DATA_21, SAMPL_DATA_21] = genTrainData (TS_COMPLETE_21, L_TS_COMPLETE_21, learningFrac, 'inorder');
            % [L_SAMPL_DATA_22, SAMPL_DATA_22] = genTrainData (TS_COMPLETE_22, L_TS_COMPLETE_22, learningFrac, 'inorder');
            % [L_SAMPL_DATA_31, SAMPL_DATA_31] = genTrainData (TS_COMPLETE_31, L_TS_COMPLETE_31, learningFrac, 'inorder');
            % [L_SAMPL_DATA_32, SAMPL_DATA_32] = genTrainData (TS_COMPLETE_32, L_TS_COMPLETE_32, learningFrac, 'inorder');
            % [L_SAMPL_DATA_41, SAMPL_DATA_41] = genTrainData (TS_COMPLETE_41, L_TS_COMPLETE_41, learningFrac, 'inorder');
            % [L_SAMPL_DATA_42, SAMPL_DATA_42] = genTrainData (TS_COMPLETE_42, L_TS_COMPLETE_42, learningFrac, 'inorder');

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

            % [~, SAMPL_DATA_11_train, ~, SAMPL_DATA_11_test] = genTrainTestData (TS_COMPLETE_11, L_TS_COMPLETE_11, learningFrac, 'inorder');
            % [~, SAMPL_DATA_12_train, ~, SAMPL_DATA_12_test] = genTrainTestData (TS_COMPLETE_12, L_TS_COMPLETE_12, learningFrac, 'inorder');
            % [~, SAMPL_DATA_21_train, ~, SAMPL_DATA_21_test] = genTrainTestData (TS_COMPLETE_21, L_TS_COMPLETE_21, learningFrac, 'inorder');
            % [~, SAMPL_DATA_22_train, ~, SAMPL_DATA_22_test] = genTrainTestData (TS_COMPLETE_22, L_TS_COMPLETE_22, learningFrac, 'inorder');
            % [~, SAMPL_DATA_31_train, ~, SAMPL_DATA_31_test] = genTrainTestData (TS_COMPLETE_31, L_TS_COMPLETE_31, learningFrac, 'inorder');
            % [~, SAMPL_DATA_32_train, ~, SAMPL_DATA_32_test] = genTrainTestData (TS_COMPLETE_32, L_TS_COMPLETE_32, learningFrac, 'inorder');
            % [~, SAMPL_DATA_41_train, ~, SAMPL_DATA_41_test] = genTrainTestData (TS_COMPLETE_41, L_TS_COMPLETE_41, learningFrac, 'inorder');
            % [~, SAMPL_DATA_42_train, ~, SAMPL_DATA_42_test] = genTrainTestData (TS_COMPLETE_42, L_TS_COMPLETE_42, learningFrac, 'inorder');


            % [L_SAMPL_DATA_03, SAMPL_DATA_03] = genTrainData (TS_COMPLETE_03, L_TS_COMPLETE_03, learningFrac, 'inorder');
            % [L_SAMPL_DATA_04, SAMPL_DATA_04] = genTrainData (TS_COMPLETE_04, L_TS_COMPLETE_04, learningFrac, 'inorder');
            % [L_SAMPL_DATA_13, SAMPL_DATA_13] = genTrainData (TS_COMPLETE_13, L_TS_COMPLETE_13, learningFrac, 'inorder');
            % [L_SAMPL_DATA_14, SAMPL_DATA_14] = genTrainData (TS_COMPLETE_14, L_TS_COMPLETE_14, learningFrac, 'inorder');
            % [L_SAMPL_DATA_23, SAMPL_DATA_23] = genTrainData (TS_COMPLETE_23, L_TS_COMPLETE_23, learningFrac, 'inorder');
            % [L_SAMPL_DATA_24, SAMPL_DATA_24] = genTrainData (TS_COMPLETE_24, L_TS_COMPLETE_24, learningFrac, 'inorder');
            % [L_SAMPL_DATA_33, SAMPL_DATA_33] = genTrainData (TS_COMPLETE_33, L_TS_COMPLETE_33, learningFrac, 'inorder');
            % [L_SAMPL_DATA_34, SAMPL_DATA_34] = genTrainData (TS_COMPLETE_34, L_TS_COMPLETE_34, learningFrac, 'inorder');
            % [L_SAMPL_DATA_43, SAMPL_DATA_43] = genTrainData (TS_COMPLETE_43, L_TS_COMPLETE_43, learningFrac, 'inorder');
            % [L_SAMPL_DATA_44, SAMPL_DATA_44] = genTrainData (TS_COMPLETE_44, L_TS_COMPLETE_44, learningFrac, 'inorder');

            reduced_SAMPL_DATA_01_train = SAMPL_DATA_01_train;
            reduced_SAMPL_DATA_01_train(reduced_SAMPL_DATA_01_train > 0) = 1;
            reduced_SAMPL_DATA_01_train(reduced_SAMPL_DATA_01_train < 0) = 2;
            reduced_SAMPL_DATA_02_train = SAMPL_DATA_02_train;
            reduced_SAMPL_DATA_02_train(reduced_SAMPL_DATA_02_train > 0) = 1;
            reduced_SAMPL_DATA_02_train(reduced_SAMPL_DATA_02_train < 0) = 2;

            reduced_SAMPL_DATA_01_test = SAMPL_DATA_01_test;
            reduced_SAMPL_DATA_01_test(reduced_SAMPL_DATA_01_test > 0) = 1;
            reduced_SAMPL_DATA_01_test(reduced_SAMPL_DATA_01_test < 0) = 2;
            reduced_SAMPL_DATA_02_test = SAMPL_DATA_02_test;
            reduced_SAMPL_DATA_02_test(reduced_SAMPL_DATA_02_test > 0) = 1;
            reduced_SAMPL_DATA_02_test(reduced_SAMPL_DATA_02_test < 0) = 2;
            % reduced_SAMPL_DATA_03 = SAMPL_DATA_03;
            % reduced_SAMPL_DATA_03(reduced_SAMPL_DATA_03 > 0) = 1;
            % reduced_SAMPL_DATA_03(reduced_SAMPL_DATA_03 < 0) = 2;
            % reduced_SAMPL_DATA_04 = SAMPL_DATA_04;
            % reduced_SAMPL_DATA_04(reduced_SAMPL_DATA_04 > 0) = 1;
            % reduced_SAMPL_DATA_04(reduced_SAMPL_DATA_04 < 0) = 2;
            reduced_SAMPL_DATA_11_train = SAMPL_DATA_11_train;
            reduced_SAMPL_DATA_11_train(reduced_SAMPL_DATA_11_train > 0) = 1;
            reduced_SAMPL_DATA_11_train(reduced_SAMPL_DATA_11_train < 0) = 2;
            reduced_SAMPL_DATA_12_train = SAMPL_DATA_12_train;
            reduced_SAMPL_DATA_12_train(reduced_SAMPL_DATA_12_train > 0) = 1;
            reduced_SAMPL_DATA_12_train(reduced_SAMPL_DATA_12_train < 0) = 2;

            reduced_SAMPL_DATA_11_test = SAMPL_DATA_11_test;
            reduced_SAMPL_DATA_11_test(reduced_SAMPL_DATA_11_test > 0) = 1;
            reduced_SAMPL_DATA_11_test(reduced_SAMPL_DATA_11_test < 0) = 2;
            reduced_SAMPL_DATA_12_test = SAMPL_DATA_12_test;
            reduced_SAMPL_DATA_12_test(reduced_SAMPL_DATA_12_test > 0) = 1;
            reduced_SAMPL_DATA_12_test(reduced_SAMPL_DATA_12_test < 0) = 2;
            % reduced_SAMPL_DATA_13 = SAMPL_DATA_13;
            % reduced_SAMPL_DATA_13(reduced_SAMPL_DATA_13 > 0) = 1;
            % reduced_SAMPL_DATA_13(reduced_SAMPL_DATA_13 < 0) = 2;
            % reduced_SAMPL_DATA_14 = SAMPL_DATA_14;
            % reduced_SAMPL_DATA_14(reduced_SAMPL_DATA_14 > 0) = 1;
            % reduced_SAMPL_DATA_14(reduced_SAMPL_DATA_14 < 0) = 2;
            reduced_SAMPL_DATA_21_train = SAMPL_DATA_21_train;
            reduced_SAMPL_DATA_21_train(reduced_SAMPL_DATA_21_train > 0) = 1;
            reduced_SAMPL_DATA_21_train(reduced_SAMPL_DATA_21_train < 0) = 2;
            reduced_SAMPL_DATA_22_train = SAMPL_DATA_22_train;
            reduced_SAMPL_DATA_22_train(reduced_SAMPL_DATA_22_train > 0) = 1;
            reduced_SAMPL_DATA_22_train(reduced_SAMPL_DATA_22_train < 0) = 2;

            reduced_SAMPL_DATA_21_test = SAMPL_DATA_21_test;
            reduced_SAMPL_DATA_21_test(reduced_SAMPL_DATA_21_test > 0) = 1;
            reduced_SAMPL_DATA_21_test(reduced_SAMPL_DATA_21_test < 0) = 2;
            reduced_SAMPL_DATA_22_test = SAMPL_DATA_22_test;
            reduced_SAMPL_DATA_22_test(reduced_SAMPL_DATA_22_test > 0) = 1;
            reduced_SAMPL_DATA_22_test(reduced_SAMPL_DATA_22_test < 0) = 2;
            % reduced_SAMPL_DATA_23 = SAMPL_DATA_23;
            % reduced_SAMPL_DATA_23(reduced_SAMPL_DATA_23 > 0) = 1;
            % reduced_SAMPL_DATA_23(reduced_SAMPL_DATA_23 < 0) = 2;
            % reduced_SAMPL_DATA_24 = SAMPL_DATA_24;
            % reduced_SAMPL_DATA_24(reduced_SAMPL_DATA_24 > 0) = 1;
            % reduced_SAMPL_DATA_24(reduced_SAMPL_DATA_24 < 0) = 2;
            reduced_SAMPL_DATA_31_train = SAMPL_DATA_31_train;
            reduced_SAMPL_DATA_31_train(reduced_SAMPL_DATA_31_train > 0) = 1;
            reduced_SAMPL_DATA_31_train(reduced_SAMPL_DATA_31_train < 0) = 2;
            reduced_SAMPL_DATA_32_train = SAMPL_DATA_32_train;
            reduced_SAMPL_DATA_32_train(reduced_SAMPL_DATA_32_train > 0) = 1;
            reduced_SAMPL_DATA_32_train(reduced_SAMPL_DATA_32_train < 0) = 2;

            reduced_SAMPL_DATA_31_test = SAMPL_DATA_31_test;
            reduced_SAMPL_DATA_31_test(reduced_SAMPL_DATA_31_test > 0) = 1;
            reduced_SAMPL_DATA_31_test(reduced_SAMPL_DATA_31_test < 0) = 2;
            reduced_SAMPL_DATA_32_test = SAMPL_DATA_32_test;
            reduced_SAMPL_DATA_32_test(reduced_SAMPL_DATA_32_test > 0) = 1;
            reduced_SAMPL_DATA_32_test(reduced_SAMPL_DATA_32_test < 0) = 2;
            % reduced_SAMPL_DATA_33 = SAMPL_DATA_33;
            % reduced_SAMPL_DATA_33(reduced_SAMPL_DATA_33 > 0) = 1;
            % reduced_SAMPL_DATA_33(reduced_SAMPL_DATA_33 < 0) = 2;
            % reduced_SAMPL_DATA_34 = SAMPL_DATA_34;
            % reduced_SAMPL_DATA_34(reduced_SAMPL_DATA_34 > 0) = 1;
            % reduced_SAMPL_DATA_34(reduced_SAMPL_DATA_34 < 0) = 2;
            reduced_SAMPL_DATA_41_train = SAMPL_DATA_41_train;
            reduced_SAMPL_DATA_41_train(reduced_SAMPL_DATA_41_train > 0) = 1;
            reduced_SAMPL_DATA_41_train(reduced_SAMPL_DATA_41_train < 0) = 2;
            reduced_SAMPL_DATA_42_train = SAMPL_DATA_42_train;
            reduced_SAMPL_DATA_42_train(reduced_SAMPL_DATA_42_train > 0) = 1;
            reduced_SAMPL_DATA_42_train(reduced_SAMPL_DATA_42_train < 0) = 2;

            reduced_SAMPL_DATA_41_test = SAMPL_DATA_41_test;
            reduced_SAMPL_DATA_41_test(reduced_SAMPL_DATA_41_test > 0) = 1;
            reduced_SAMPL_DATA_41_test(reduced_SAMPL_DATA_41_test < 0) = 2;
            reduced_SAMPL_DATA_42_test = SAMPL_DATA_42_test;
            reduced_SAMPL_DATA_42_test(reduced_SAMPL_DATA_42_test > 0) = 1;
            reduced_SAMPL_DATA_42_test(reduced_SAMPL_DATA_42_test < 0) = 2;
            % reduced_SAMPL_DATA_43 = SAMPL_DATA_43;
            % reduced_SAMPL_DATA_43(reduced_SAMPL_DATA_43 > 0) = 1;
            % reduced_SAMPL_DATA_43(reduced_SAMPL_DATA_43 < 0) = 2;
            % reduced_SAMPL_DATA_44 = SAMPL_DATA_44;
            % reduced_SAMPL_DATA_44(reduced_SAMPL_DATA_44 > 0) = 1;
            % reduced_SAMPL_DATA_44(reduced_SAMPL_DATA_44 < 0) = 2;

            reduced_L_SAMPL_DATA_1_train = L_SAMPL_DATA_01_train - 1;
            reduced_L_SAMPL_DATA_2_train = L_SAMPL_DATA_02_train - 1;

            reduced_L_SAMPL_DATA_1_test = L_SAMPL_DATA_01_test - 1;
            reduced_L_SAMPL_DATA_2_test = L_SAMPL_DATA_02_test - 1;
            % reduced_L_SAMPL_DATA_3 = L_SAMPL_DATA_03 - 1;
            % reduced_L_SAMPL_DATA_4 = L_SAMPL_DATA_04 - 1;

            %Sparse biopolar mapping
            %creates matrix of random hypervectors with element values 1, 0, and -1,
            %matrix is has feature (channel) numbers of binary D size hypervectors
            %Should be the S vectors
            q=0.7;

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

            projM1=projBRandomHV(D,channels_v,q);
            projM3=projBRandomHV(D,channels_v_EEG,q);
            projM5=projBRandomHV(D,channels_v_GSR,q);
            projM7=projBRandomHV(D,channels_v_BVP,q);
            projM9=projBRandomHV(D,channels_v_RES,q);

            projM1_neg = projM1;
            projM1_pos = projM1;
            projM1_neg(projM1_neg==-1) = 0;
            projM1_pos(projM1_pos==1) = 0;
            projM1_pos(projM1==-1) = 1;
            projM3_neg = projM3;
            projM3_pos = projM3;
            projM3_neg(projM3==-1) = 0;
            projM3_pos(projM3==1) = 0;
            projM3_pos(projM3==-1) = 1;
            projM5_neg = projM5;
            projM5_pos = projM5;
            projM5_neg(projM5==-1) = 0;
            projM5_pos(projM5==1) = 0;
            projM5_pos(projM5==-1) = 1;
            projM7_neg = projM7;
            projM7_pos = projM7;
            projM7_neg(projM7==-1) = 0;
            projM7_pos(projM7==1) = 0;
            projM7_pos(projM7==-1) = 1;
            projM9_neg = projM9;
            projM9_pos = projM9;
            projM9_neg(projM9==-1) = 0;
            projM9_pos(projM9==1) = 0;
            projM9_pos(projM9==-1) = 1;


            for N = ngram:ngram
                % creates ngram for data, rotates through and 
                N

                %% L
                % generate ngram bundles for each data stream

                % fprintf ('HDC for L\n');
                % if (select == 1)
                %     [numpat, hdc_model_2] = hdctrainproj (classes, reduced_L_SAMPL_DATA_4, reduced_L_SAMPL_DATA_4, reduced_L_SAMPL_DATA_4, reduced_L_SAMPL_DATA_4, reduced_L_SAMPL_DATA_4,reduced_SAMPL_DATA_04, reduced_SAMPL_DATA_14, reduced_SAMPL_DATA_24, reduced_SAMPL_DATA_34, reduced_SAMPL_DATA_44, chAM8, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, channels_l, channels_l_EEG, channels_l_GSR, channels_l_BVP, channels_l_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);
                % else
                %     [numpat_2, hdc_model_2] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_2, chAM8, iMch1, D, N, precision, channels_a,projM1_pos,projM1_neg, classes); 
                %     [numpat_4, hdc_model_4] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_4, chAM8, iMch3, D, N, precision, channels_a_ECG,projM3_pos,projM3_neg, classes); 
                %     [numpat_6, hdc_model_6] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_6, chAM8, iMch5, D, N, precision, channels_a_EEG,projM5_pos,projM5_neg, classes); 
                % end
                % 
                % %bundle all the sensors (this is the fusion point)
                % if (select ~= 1)
                %     %class 1
                %     hdc_model_2(0)=mode([hdc_model_2(0); hdc_model_4(0); hdc_model_6(0)]);
                %     %class 2
                %     hdc_model_2(1)=mode([hdc_model_2(1); hdc_model_4(1); hdc_model_6(1)]);
                % end
                % 
                % [acc_ex2, acc2, pl2, al2, all_error] = hdcpredictproj  (reduced_L_TS_COMPLETE_4, reduced_TS_COMPLETE_04, reduced_L_TS_COMPLETE_4, reduced_TS_COMPLETE_14, reduced_L_TS_COMPLETE_4, reduced_TS_COMPLETE_24, reduced_L_TS_COMPLETE_4, reduced_TS_COMPLETE_34, reduced_L_TS_COMPLETE_4, reduced_TS_COMPLETE_44, hdc_model_2, chAM8, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, classes, channels_l, channels_l_EEG, channels_l_GSR, channels_l_BVP, channels_l_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);
                % 
                % accuracy(N,2) = acc2;
                % acc2
                % acc_matrix((randCounter*4),(D/1000)) = acc2;
                % acc_matrix_L(randCounter,(D/1000)) = acc2;
                % 
                % %acc_ngram_1(N,j)=acc1;
                % acc_ngram_L(N,j)=acc2;

                %% D
                % 
                % fprintf ('HDC for D\n');
                % if (select == 1)
                %     [numpat, hdc_model_2] = hdctrainproj (classes, reduced_L_SAMPL_DATA_3, reduced_L_SAMPL_DATA_3, reduced_L_SAMPL_DATA_3, reduced_L_SAMPL_DATA_3, reduced_L_SAMPL_DATA_3,reduced_SAMPL_DATA_03, reduced_SAMPL_DATA_13, reduced_SAMPL_DATA_23, reduced_SAMPL_DATA_33, reduced_SAMPL_DATA_43, chAM8, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, channels_d, channels_d_EEG, channels_d_GSR, channels_d_BVP, channels_d_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);
                % else
                %     [numpat_2, hdc_model_2] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_2, chAM8, iMch1, D, N, precision, channels_a,projM1_pos,projM1_neg, classes); 
                %     [numpat_4, hdc_model_4] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_4, chAM8, iMch3, D, N, precision, channels_a_ECG,projM3_pos,projM3_neg, classes); 
                %     [numpat_6, hdc_model_6] = hdctrainproj (reduced_L_SAMPL_DATA_2, reduced_SAMPL_DATA_6, chAM8, iMch5, D, N, precision, channels_a_EEG,projM5_pos,projM5_neg, classes); 
                % end
                % 
                % %bundle all the sensors (this is the fusion point)
                % if (select ~= 1)
                %     %class 1
                %     hdc_model_2(0)=mode([hdc_model_2(0); hdc_model_4(0); hdc_model_6(0)]);
                %     %class 2
                %     hdc_model_2(1)=mode([hdc_model_2(1); hdc_model_4(1); hdc_model_6(1)]);
                % end
                % 
                % [acc_ex2, acc2, pl2, al2, all_error] = hdcpredictproj  (reduced_L_TS_COMPLETE_3, reduced_TS_COMPLETE_03, reduced_L_TS_COMPLETE_3, reduced_TS_COMPLETE_13, reduced_L_TS_COMPLETE_3, reduced_TS_COMPLETE_23, reduced_L_TS_COMPLETE_3, reduced_TS_COMPLETE_33, reduced_L_TS_COMPLETE_3, reduced_TS_COMPLETE_43, hdc_model_2, chAM8, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, classes, channels_d, channels_d_EEG, channels_d_GSR, channels_d_BVP, channels_d_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);
                % 
                % accuracy(N,2) = acc2;
                % acc2
                % acc_matrix((randCounter*4 - 1),(D/1000)) = acc2;
                % acc_matrix_D(randCounter,(D/1000)) = acc2;
                % 
                % acc_ngram_D(N,j)=acc2;

                %% V
                randCounter
                fprintf ('HDC for V\n');
                if (select == 1)
                    [hdc_model_2] = hdctrainproj (classes, reduced_L_SAMPL_DATA_2_train, reduced_SAMPL_DATA_02_train, reduced_SAMPL_DATA_12_train, reduced_SAMPL_DATA_22_train, reduced_SAMPL_DATA_32_train, reduced_SAMPL_DATA_42_train, chAM8, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, channels_v, channels_v_EEG, channels_v_GSR, channels_v_BVP, channels_v_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);
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

                %[acc_ex2v, acc2, pl2v, al2v, all_error,tranzErrorv] = hdcpredictproj  (reduced_L_TS_COMPLETE_2, reduced_TS_COMPLETE_02, reduced_TS_COMPLETE_12, reduced_TS_COMPLETE_22, reduced_TS_COMPLETE_32, reduced_TS_COMPLETE_42, hdc_model_2, chAM8, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, classes, channels_v, channels_v_EEG, channels_v_GSR, channels_v_BVP, channels_v_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);
                [accexc_alltrz_v, acc_ex2v, acc2, pl2v, al2v, all_error,tranzErrorv] = hdcpredictproj  (reduced_L_SAMPL_DATA_2_test, reduced_SAMPL_DATA_02_test, reduced_SAMPL_DATA_12_test, reduced_SAMPL_DATA_22_test, reduced_SAMPL_DATA_32_test, reduced_SAMPL_DATA_42_test, hdc_model_2, chAM8, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, classes, channels_v, channels_v_EEG, channels_v_GSR, channels_v_BVP, channels_v_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);

                pl2v = pl2v';
                %accuracy(N,2) = acc2;
                acc2
                acc_matrix(((randCounter)*2),subject) = acc2;
                acc_matrix_V(randCounter,subject) = acc2;
                acc_matrix_Vex(randCounter,subject) = acc_ex2v;
                acc_matrix_Vex_all(randCounter,subject) = accexc_alltrz_v;
                acc_hamclass_V(randCounter,1) = sum(xor(hdc_model_2(0),hdc_model_2(1)));
                prediction_values_V(1:length(L_SAMPL_DATA_02_test),randCounter) = pl2v;
                actual_values_V(1:length(L_SAMPL_DATA_02_test),randCounter) = al2v;
                val_right = (pl2v == al2v);

                %acc_ngram_V(N,j)=acc2;

                %% A
                randCounter
                fprintf ('HDC for A\n');
                if (select == 1)
                    [hdc_model_2] = hdctrainproj (classes, reduced_L_SAMPL_DATA_1_train, reduced_SAMPL_DATA_01_train, reduced_SAMPL_DATA_11_train, reduced_SAMPL_DATA_21_train, reduced_SAMPL_DATA_31_train, reduced_SAMPL_DATA_41_train, chAM8, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, channels_a, channels_a_EEG, channels_a_GSR, channels_a_BVP, channels_a_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);
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

                %[acc_ex2a, acc2, pl2a, al2a, all_error,tranzErrora] = hdcpredictproj  (reduced_L_TS_COMPLETE_1, reduced_TS_COMPLETE_01, reduced_TS_COMPLETE_11, reduced_TS_COMPLETE_21, reduced_TS_COMPLETE_31, reduced_TS_COMPLETE_41, hdc_model_2, chAM8, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, classes, channels_a, channels_a_EEG, channels_a_GSR, channels_a_BVP, channels_a_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);
                [accexc_alltrz_a, acc_ex2a, acc2, pl2a, al2a, all_error,tranzErrora] = hdcpredictproj  (reduced_L_SAMPL_DATA_1_test, reduced_SAMPL_DATA_01_test, reduced_SAMPL_DATA_11_test, reduced_SAMPL_DATA_21_test, reduced_SAMPL_DATA_31_test, reduced_SAMPL_DATA_41_test, hdc_model_2, chAM8, iMch1, iMch3, iMch5, iMch7, iMch9, D, N, precision, classes, channels_a, channels_a_EEG, channels_a_GSR, channels_a_BVP, channels_a_RES, projM1_pos, projM1_neg, projM3_pos, projM3_neg, projM5_pos, projM5_neg, projM7_pos, projM7_neg, projM9_pos, projM9_neg);

                pl2a = pl2a';
                %accuracy(N,2) = acc2;
                acc2
                acc_matrix(((randCounter)*2-1),subject) = acc2;
                acc_matrix_A(randCounter,subject) = acc2;
                acc_matrix_Aex(randCounter,subject) = acc_ex2a;
                acc_matrix_Aex_all(randCounter,subject) = accexc_alltrz_a;
                acc_hamclass_A(randCounter,1) = sum(xor(hdc_model_2(0),hdc_model_2(1)));
                ar_right = (pl2a == al2a);
                prediction_values_A(1:length(L_SAMPL_DATA_01_test),randCounter) = pl2a;
                actual_values_A(1:length(L_SAMPL_DATA_01_test),randCounter) = al2a;

                %acc_ngram_1(N,j)=acc1;
                %acc_ngram_A(N,j)=acc2;
            end

        end

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

        randCounter=randCounter-1;
    end
    prediction_values_V(length(L_SAMPL_DATA_02_test)+1:size(features,1),:)=[];
    prediction_values_A(length(L_SAMPL_DATA_01_test)+1:size(features,1),:)=[];
    actual_values_A(length(L_SAMPL_DATA_01_test)+1:size(features,1),:)=[];
    actual_values_V(length(L_SAMPL_DATA_02_test)+1:size(features,1),:)=[];
    randCounter = full_count;
end
acc_matrix;
acc_av_A_matrix = mean(acc_matrix_A);
acc_av_V_matrix = mean(acc_matrix_V);
acc_OverallAav = mean(acc_av_A_matrix)
acc_OverallVav = mean(acc_av_V_matrix)
acc_max_A_matrix = max(acc_matrix_A);
acc_max_V_matrix = max(acc_matrix_V);
acc_OverallAmax = mean(acc_max_A_matrix);
acc_OverallVmax = mean(acc_max_V_matrix);

acc_av_ex_A_matrix = mean(acc_matrix_Aex);
acc_av_ex_V_matrix = mean(acc_matrix_Vex);
acc_OverallAav_ex = mean(acc_av_ex_A_matrix)
acc_OverallVav_ex = mean(acc_av_ex_V_matrix)
acc_max_ex_A_matrix = max(acc_matrix_Aex);
acc_max_ex_V_matrix = max(acc_matrix_Vex);
acc_OverallAmax_ex = mean(acc_max_ex_A_matrix);
acc_OverallVmax_ex = mean(acc_max_ex_V_matrix);

acc_av_ex_all_A_matrix = mean(acc_matrix_Aex_all);
acc_av_ex_all_V_matrix = mean(acc_matrix_Vex_all);
acc_OverallAav_ex_all = mean(acc_av_ex_all_A_matrix)
acc_OverallVav_ex_all = mean(acc_av_ex_all_V_matrix)
acc_max_ex_all_A_matrix = max(acc_matrix_Aex_all);
acc_max_ex_all_V_matrix = max(acc_matrix_Vex_all);
acc_OverallAmax_ex_all = mean(acc_max_ex_all_A_matrix);
acc_OverallVmax_ex_all = mean(acc_max_ex_all_V_matrix);