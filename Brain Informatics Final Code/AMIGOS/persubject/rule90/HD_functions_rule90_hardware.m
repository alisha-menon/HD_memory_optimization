
function message = HD_functions_rule90_hardware
  assignin('base','arrange_vectors', @arrange_vectors); 
  assignin('base','genBRandomHV', @genBRandomHV); 
  assignin('base','final_arrange', @final_arrange); 
  assignin('base','vector_counter', @vector_counter); 
  assignin('base','projBRandomHV', @projBRandomHV); 
  assignin('base','projRandomHV', @projRandomHV); 
  assignin('base','initItemMemories', @initItemMemories);
  assignin('base','projItemMemeory', @projItemMemeory); 
  assignin('base','computeNgramproj', @computeNgramproj); 
  assignin('base','hdctrainproj', @hdctrainproj); 
  assignin('base','hdcpredictproj', @hdcpredictproj);   
  assignin('base','genRandomHV', @genRandomHV); 
  assignin('base','downSampling', @downSampling);
  assignin('base','genTrainData', @genTrainData);
  assignin('base','genTrainTestData', @genTrainTestData);
  assignin('base','select_traintest', @select_traintest);
  assignin('base','lookupItemMemeory', @lookupItemMemeory);
  assignin('base','hamming', @hamming);
  assignin('base', 'foldedrule90', @foldedrule90);
  message='Importing all HD functions';
end

function [L1, L2, L_SAMPL_DATA_train, SAMPL_DATA_train, L_SAMPL_DATA_test, SAMPL_DATA_test] = genTrainTestData (data, labels, trainingFrac, order,N, subject)
%
% DESCRIPTION   : generates a dataset to train the alorithm using a fraction of the input data 
%
% INPUTS:
%   data        : input data
%   labels      : input labels
%   trainingFrac: the fraction of data we should use to output a training dataset
%   order       : whether preserve the order of inputs (inorder) or randomly select
%   donwSampRate: the rate or stride of downsampling
% OUTPUTS:
%   SAMPL_DATA  : dataset for training
%   L_SAMPL_DATA: corresponding labels
%    

	%rng('default');
    %rng(1);
    size = length(labels);
    L1 = 1:size;
    L2 = L1((subject-1)*(size/33)+1:subject*(size/33));
    L1(L2) = [];
    
    a = find (labels(L1) == 1);
    b = find (labels(L1) == 2);

%     L1 = L1(randperm(length(L1)));
%     L2 = L2(randperm(length(L2)));
    
    L_SAMPL_DATA_train = labels (L1(a));
    L_SAMPL_DATA_train = [L_SAMPL_DATA_train; (labels(L1(b)))];
    SAMPL_DATA_train = data (L1(a),:);
    SAMPL_DATA_train = [SAMPL_DATA_train; (data(L1(b),:))];
    
    SAMPL_DATA_test = data(L2,:);
    L_SAMPL_DATA_test = labels(L2);
    
    L1 = a;
    L2 = b;
end

function [sample_data_train, sample_data_test] = select_traintest(L1, L2, data)
    sample_data_train = [data(L1,:); data(L2,:)];
    sample_data_test = data;
    sample_data_test([L1;L2],:) = [];
    
end

function vec_array = arrange_vectors(m)
    vec_array = [];
    m_copy = m;
    if (mod(length(m_copy), 2)== 0)
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

function [L_SAMPL_DATA, SAMPL_DATA] = genTrainData (data, labels, trainingFrac, order)
%
% DESCRIPTION   : generates a dataset to train the alorithm using a fraction of the input data 
%
% INPUTS:
%   data        : input data
%   labels      : input labels
%   trainingFrac: the fraction of data we should use to output a training dataset
%   order       : whether preserve the order of inputs (inorder) or randomly select
%   donwSampRate: the rate or stride of downsampling
% OUTPUTS:
%   SAMPL_DATA  : dataset for training
%   L_SAMPL_DATA: corresponding labels
%    

	rng('default');
    rng(1);
    L1 = find (labels == 1);
    L2 = find (labels == 2);
    L3 = find (labels == 3);
    L4 = find (labels == 4);
    L5 = find (labels == 5);
	L6 = find (labels == 6);
	L7 = find (labels == 7);
   
    L1 = L1 (1 : floor(length(L1) * trainingFrac));
    L2 = L2 (1 : floor(length(L2) * trainingFrac));
    L3 = L3 (1 : floor(length(L3) * trainingFrac));
    L4 = L4 (1 : floor(length(L4) * trainingFrac));
    L5 = L5 (1 : floor(length(L5) * trainingFrac));
	L6 = L6 (1 : floor(length(L6) * trainingFrac));
	L7 = L7 (1 : floor(length(L7) * trainingFrac));
 
    if order == 'inorder'
		Inx1 = 1:1:length(L1);
		Inx2 = 1:1:length(L2);
		Inx3 = 1:1:length(L3);
		Inx4 = 1:1:length(L4);
		Inx5 = 1:1:length(L5);
		Inx6 = 1:1:length(L6);
		Inx7 = 1:1:length(L7);
	else
		Inx1 = randperm (length(L1));
		Inx2 = randperm (length(L2));
		Inx3 = randperm (length(L3));
		Inx4 = randperm (length(L4));
		Inx5 = randperm (length(L5));
		Inx6 = randperm (length(L6));
		Inx7 = randperm (length(L7));
	end
    
    L_SAMPL_DATA = labels (L1(Inx1));
    L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L2(Inx2)))];
    L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L3(Inx3)))];
    L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L4(Inx4)))];
    L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L5(Inx5)))];
	L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L6(Inx6)))];
	L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L7(Inx7)))];
    %L_SAMPL_DATA = L_SAMPL_DATA';
    
    SAMPL_DATA   = data (L1(Inx1), :);
    SAMPL_DATA   = [SAMPL_DATA; (data(L2(Inx2), :))];
    SAMPL_DATA   = [SAMPL_DATA; (data(L3(Inx3), :))];
    SAMPL_DATA   = [SAMPL_DATA; (data(L4(Inx4), :))];
    SAMPL_DATA   = [SAMPL_DATA; (data(L5(Inx5), :))];
	SAMPL_DATA   = [SAMPL_DATA; (data(L6(Inx6), :))];
	SAMPL_DATA   = [SAMPL_DATA; (data(L7(Inx7), :))];
end

function [CiM, iM] = initItemMemories (D, MAXL, channels)
%
% DESCRIPTION   : initialize the item Memory  
%
% INPUTS:
%   D           : Dimension of vectors
%   MAXL        : Maximum amplitude of EMG signal
%   channels    : Number of acquisition channels
% OUTPUTS:
%   iM          : item memory for IDs of channels
%   CiM         : continious item memory for value of a channel
 
    % MAXL = 21;
	CiM = containers.Map ('KeyType','double','ValueType','any');
	iM  = containers.Map ('KeyType','double','ValueType','any');
    rng('shuffle');
      
    for i = 1 : channels
        iM(i) = genRandomHV (D);
    end

    initHV = genRandomHV (D);
	currentHV = initHV;
	randomIndex = randperm (D);
	
    for i = 0:1:MAXL
        CiM(i) = currentHV; 
        SP = floor(D/2/MAXL);
		startInx = (i*SP) + 1;
		endInx = ((i+1)*SP) + 1;
		currentHV (randomIndex(startInx : endInx)) = not(currentHV (randomIndex(startInx: endInx)));
    end
end

function randomHV = lookupItemMemeory (itemMemory, rawKey, precision)
%
% DESCRIPTION   : recalls a vector from item Memory based on inputs
%
% INPUTS:
%   itemMemory  : item memory
%   rawKey      : the input key
%   D           : Dimension of vectors
%   precision   : precision used in quantization of input EMG signals
%
% OUTPUTS:
%   randomHV    : return the related vector

 
    key = int64 (rawKey * precision);
  
    if itemMemory.isKey (key) 
        randomHV = itemMemory (key);
    else
        fprintf ('CANNOT FIND THIS KEY: %d\n', key);       
    end
end

function randomHV = genRandomHV(D)
%
% DESCRIPTION   : generate a random vector with zero mean 
%
% INPUTS:
%   D           : Dimension of vectors
% OUTPUTS:
%   randomHV    : generated random vector
    rng('shuffle');
    if mod(D,2)
        disp ('Dimension is odd!!');
    else
        randomIndex = randperm (D);
        randomHV (randomIndex(1 : D/2)) = 1;
        randomHV (randomIndex(D/2+1 : D)) = 0;
       
    end
end

function [downSampledData, downSampledLabels] = downSampling (data, labels, donwSampRate)
%
% DESCRIPTION   : apply a downsampling to get rid of redundancy in signals 
%
% INPUTS:
%   data        : input data
%   labels      : input labels
%   donwSampRate: the rate or stride of downsampling
% OUTPUTS:
%   downSampledData: downsampled data
%   downSampledLabels: downsampled labels
%    
	j = 1;
    
    for i = 1:donwSampRate:length(data(:,1))
        
		downSampledData (j,:) = data(i, :);
		downSampledLabels (j) = labels(i);
        j = j + 1;
        
    end
    
    downSampledLabels = downSampledLabels';
    
end
	
function proj_m = projBRandomHV( D, F ,q)
%   D: dim
%   F: number of features
%   q: sparsity

   proj_m=[]; 
   if mod(D,2)
        disp ('Dimension is odd!!');
   else   
    F_D=F*D;
    probM=rand(F,D);
    p_n1=(1-q)/2;
    p_p1=p_n1;

    for k=1:F
     for i=1:D
         if probM(k,i)<p_n1     
            proj_m(k,i)=-1;
         else if (p_n1<=probM(k,i)) && (probM(k,i)<(q+p_n1))
           proj_m(k,i)=0;   
         else 
           proj_m(k,i)=1; 
         end
     end
     end 
    end
   end
 
end

function proj_m = projRandomHV(D, F)
%   D: dim
%   F: number of features
%   q: sparsity
   rng('shuffle');
   proj_m=zeros(F,D); 
   %proj_m = [];
   if mod(D,2)
        disp ('Dimension is odd!!');
   else   
        %F_D=F*D;
        %probM=rand(F,D);
        %p_n1=(1-q)/2;
        %p_p1=p_n1;
       for k=1:F
           proj_m(k,:) = genRandomHV (D);
       end 
   end
end

function randomHV = projItemMemeory (projM_pos, projM_neg, voffeature,ioffeature,D)
%
% INPUTS:
%   projM	: random vector with {-1,0,+1}
%   voffeature	: value of a feature
%   ioffeature	: index of a feature
% OUTPUTS:
%   randomHV    : return the related vector

        if (voffeature > 0)
            randomHV = projM_pos(ioffeature,:);
        elseif (voffeature < 0)
            randomHV = projM_neg(ioffeature,:);
        else
            randomHV = zeros(1,D);
        end
        %projV=projM(ioffeature,:);
        %h= voffeature.*projV;

    %for i=1:length(h)
      %if h(i)>0
        %randomHV(i)=1;
        %else
        %randomHV(i)=0;
       %end
    %end

end

function randomHV = projItemMemeory_bin (projM_pos, projM_neg, voffeature,ioffeature,D)
%
% INPUTS:
%   projM	: random vector with {-1,0,+1}
%   voffeature	: value of a feature
%   ioffeature	: index of a feature
% OUTPUTS:
%   randomHV    : return the related vector

        if (voffeature == 1)
            randomHV = projM_pos(ioffeature,:);
        elseif (voffeature == 2)
            randomHV = projM_neg(ioffeature,:);
        else
            randomHV = zeros(1,D);
        end
        %projV=projM(ioffeature,:);
        %h= voffeature.*projV;

    %for i=1:length(h)
      %if h(i)>0
        %randomHV(i)=1;
        %else
        %randomHV(i)=0;
       %end
    %end

end

function v = computeNgramproj (buffer, CiM, N, precision, iM, channels,projM_pos, projM_neg, sample_index,D)
% 	DESCRIPTION: computes the N-gram
% 	INPUTS:
% 	buffer   :  data input
% 	iM       :  Item Memory for IDs of the channels
%   N        :  dimension of the N-gram
%   precision:  precision used in quantization (no use)
% 	CiM      :  Continious Item Memory for the values of a channel (no use)
%   channels :  numeber of features
% 	OUTPUTS:
% 	Ngram    :  query hypervector
    
    %setup for first sample duplicate for all modalities
    chHV = projItemMemeory_bin (projM_pos, projM_neg, buffer(sample_index, 1),1,D);
    chHV = xor(chHV , iM(1));
    v = chHV;
    if channels>1    
    for i = 2 : channels
        chHV = projItemMemeory_bin (projM_pos, projM_neg, buffer(sample_index, i), i,D);
        chHV = xor(chHV , iM(i));
        if i == 2
            ch2HV=chHV; 
        end
        %create matrix of all channel final outputs to be bundled
        v = [v; chHV];
    end
    
    %don't understand why adding the xor of the final and 2nd channel
    chHV = xor(chHV , ch2HV);
    v = [v; chHV]; 
    end
    
%     %Add the other modalities
%     if channels==1
%     Ngram = v;
%     else
%     Ngram = mode(v);
%     end
%     
%     %Add later samples
%     for sample_index = 2:1:N
%         %replicate for other modalities
%         chHV = projItemMemeory (projM, buffer(sample_index, 1), 1);
%         chHV = xor(chHV , iM(1));
%         ch1HV = chHV;
%         %combine the other modalities
%         v = chHV;
%       if channels>1  
%         %replicate for other modalities
%         for j = 2 : channels
%             chHV = projItemMemeory (projM, buffer(sample_index, j), j);
%             chHV = xor(chHV , iM(j));
%             if j == 2
%                 ch2HV=chHV; 
%             end
%             %combine the other modalities
%             v = [v; chHV];
%         end  
%         %combine other modalities for whatever this weird thing is
%         chHV = xor(chHV , ch2HV);
%         v = [v; chHV]; 
%       end
%       
%       if channels==1
%         record = v;          
%       else
%         record = mode(v); 
%       end
% 		Ngram = xor(circshift (Ngram, [1,1]) , record);
%            
%     end	 
%  
end
  
function [numPat, AM] = hdctrainproj (classes,labelTrainSet1, labelTrainSet2, labelTrainSet3, trainSet1, trainSet2, trainSet3, CiM, iM1, iM2, iM3, D, N, precision, channels1, channels2, channels3,projM1_pos, projM1_neg, projM2_pos, projM2_neg, projM3_pos, projM3_neg ) 
%
% DESCRIPTION   : train an associative memory based on input training data
%
% INPUTS:
%   labelTrainSet : training labels
%   trainSet    : training data
%   CiM         : cont. item memory (no use)
%   iM          : item memory
%   D           : Dimension of vectors
%   N           : size of n-gram, i.e., window size
%   precision   : precision used in quantization (no use)
%
% OUTPUTS:
%   AM          : Trained associative memory
%   numPat      : Number of stored patterns for each class of AM
%
 
	AM = containers.Map ('KeyType','double','ValueType','any');

	am_index = 0;
    for label = 1:1:max(labelTrainSet1)
    	AM (am_index) = zeros (1,D);
	    numPat (label) = 0;
        am_index = am_index+1;
    end
    trainVecList=zeros (1,D);
    i = 1;
    label = labelTrainSet1 (1);
    
    while i < length(labelTrainSet1)-N+1
        if labelTrainSet1(i+N-1) == label 
        %creates ngram for label    
        %instead want to compute ngram which is fused, keep going if all
        %of the labels for the modalities are the same. Once one changes,
        %stop bundling and move onto the next sample for the next label for
        %all modalities
        %setup first Ngram
	    v1 = computeNgramproj (trainSet1 (i : i+N-1,:), CiM, N, precision, iM1, channels1,projM1_pos, projM1_neg, 1,D);
        v2 = computeNgramproj (trainSet2 (i : i+N-1,:), CiM, N, precision, iM2, channels2,projM2_pos, projM2_neg, 1,D);
        v3 = computeNgramproj (trainSet3 (i : i+N-1,:), CiM, N, precision, iM3, channels3,projM3_pos, projM3_neg, 1,D);
        if channels1==1
            ngram1 = v1;
        else
            ngram1 = mode(v1);
        end
        if channels2==1
            ngram2 = v2;
        else
            ngram2 = mode(v2);
        end
        if channels3==1
            ngram3 = v3;
        else
            ngram3 = mode(v3);
        end
        ngram=mode([ngram1;ngram2;ngram3]);
        for sample_index = 2:1:N
            v1 = computeNgramproj (trainSet1 (i : i+N-1,:), CiM, N, precision, iM1, channels1,projM1_pos, projM1_neg, sample_index,D);
            if channels1==1
                record1 = v1;          
            else
                record1 = mode(v1); 
            end
            v2 = computeNgramproj (trainSet2 (i : i+N-1,:), CiM, N, precision, iM2, channels2,projM2_pos, projM2_neg, sample_index,D);
            if channels2==1
                record2 = v2;          
            else
                record2 = mode(v2); 
            end
            v3 = computeNgramproj (trainSet3 (i : i+N-1,:), CiM, N, precision, iM3, channels3,projM3_pos, projM3_neg, sample_index,D);
            if channels3==1
                record3 = v3;          
            else
                record3 = mode(v3); 
            end
            record = mode([record1;record2;record3]);
            circ = circshift (ngram, [1,1]);
            circ(1) = 0;
            ngram = xor(circ , record);
        end
            trainVecList = [trainVecList ; ngram];
	        %numPat (labelTrainSet1 (i+N-1)) = numPat (labelTrainSet1 (i+N-1)) + 1;
            
            i = i + 1;
        else
            %once you reach the end of that label, do majority count and
            %set AM vector to be the bundled version of all the ngrams for
            %that label, then move onto next label
            trainVecList(1 , :) = 3;
            label
            AM (label) = mode (trainVecList);
            label = labelTrainSet1(i+N-1);
            i = i+N-1;
            %numPat (label) = 0;
            trainVecList=zeros (1,D);
        end
    end
    l=floor(i+(N/2));
    if l > length(labelTrainSet1)
       l= length(labelTrainSet1);
    end    
    %wrap up the last training class
    labelTrainSet1(l)
    AM (labelTrainSet1 (l)) = mode (trainVecList); 
    am_index = 0;
    for label = 1:1:classes
		fprintf ('Class = %d \t sum = %.0f \t created \n', label, sum(AM(am_index)));
        am_index = am_index + 1;
    end
end

function [accexc_alltrz, accuracy, predicLabel, actualLabel, all_error] = hdcpredictproj (labelTestSet1, testSet1, labelTestSet2, testSet2,labelTestSet3, testSet3,AM, CiM, iM1, iM2, iM3, D, N, precision, classes, channels1, channels2, channels3,projM1_pos, projM1_neg, projM2_pos, projM2_neg, projM3_pos, projM3_neg)
%
% DESCRIPTION   : test accuracy based on input testing data
%
% INPUTS:
%   labelTestSet: testing labels
%   testSet     : EMG test data
%   AM          : Trained associative memory
%   CiM         : Cont. item memory (no use)
%   iM          : item memory
%   D           : Dimension of vectors
%   N           : size of n-gram, i.e., window size 
%   precision   : precision used in quantization (no use)
%
% OUTPUTS:
%   accuracy    : classification accuracy for all situations
%   accExcTrnz  : classification accuracy excluding the transitions between gestutes
%
	correct = 0;
    numTests = 0;
	tranzError = 0;
	all_error = 0;
    second_error_all=0;
    label_am = 0;
    correctex = 0;
    numtestex = 0;
   
    for i = 1:1:size(testSet1,1)-N+1        
		numTests = numTests + 1;
		actualLabel(i : i+N-1,:) = mode(labelTestSet1 (i : i+N-1));
        %% setup first Ngram for all modalities
        v1 = computeNgramproj (testSet1 (i : i+N-1,:), CiM, N, precision, iM1, channels1,projM1_pos, projM1_neg,1,D);
        v2 = computeNgramproj (testSet2 (i : i+N-1,:), CiM, N, precision, iM2, channels2,projM2_pos, projM2_neg,1,D);
        v3 = computeNgramproj (testSet3 (i : i+N-1,:), CiM, N, precision, iM3, channels3,projM3_pos, projM3_neg,1,D);
        if channels1==1
            sigHV1 = v1;
        else
            sigHV1 = mode(v1);
        end
        if channels2==1
            sigHV2 = v2;
        else
            sigHV2 = mode(v2);
        end
        if channels3==1
            sigHV3 = v3;
        else
            sigHV3 = mode(v3);
        end
        % combine modalities
        sigHV=mode([sigHV1;sigHV2;sigHV3]);
        %% setup spatial encoder outputs for all channels for all modalities N-1 samples
        for sample_index = 2:1:N
            v1 = computeNgramproj (testSet1 (i : i+N-1,:), CiM, N, precision, iM1, channels1,projM1_pos, projM1_neg, sample_index,D);
            if channels1==1
                record1 = v1;          
            else
                record1 = mode(v1); 
            end
            v2 = computeNgramproj (testSet2 (i : i+N-1,:), CiM, N, precision, iM2, channels2,projM2_pos, projM2_neg, sample_index,D);
            if channels2==1
                record2 = v2;          
            else
                record2 = mode(v2); 
            end
            v3 = computeNgramproj (testSet3 (i : i+N-1,:), CiM, N, precision, iM3, channels3,projM3_pos, projM3_neg, sample_index,D);
            if channels3==1
                record3 = v3;          
            else
                record3 = mode(v3); 
            end
            % combine modalities
            record = mode([record1;record2;record3]);
            % bundle
            circ = circshift (sigHV, [1,1]);
            circ(1) = 0;
            sigHV = xor(circ , record);
            %sigHV = xor(circshift (sigHV, [1,1]) , record);
        end
   
	    [predict_hamm, error, second_error] = hamming(sigHV, AM, classes);
        predicLabel(i : i+N-1) = predict_hamm;
        all_error = [all_error error]; %#ok<AGROW>
        second_error_all = [second_error_all second_error]; %#ok<AGROW>
        if predict_hamm == actualLabel(i)
			correct = correct + 1;
        elseif labelTestSet1 (i) ~= labelTestSet1(i+N-1)
			tranzError = tranzError + 1;
        end
        
        if (mode(labelTestSet1(i:i+N-1)) == labelTestSet1(i))
            if predict_hamm == actualLabel(i)
                correctex = correctex+1;
            end
            numtestex = numtestex + 1;
        end
    end
    
    accuracy = correct / numTests;
    accExcTrnz = correct / (numTests - tranzError);

    accuracy = correct / numTests;
    if (numtestex == 0)
        accexc_alltrz = accuracy;
    else
        accexc_alltrz = correctex/numtestex;
    end
end

function [predict_hamm, error, second_error] = hamming (q, aM, classes)
%
% DESCRIPTION       : computes the Hamming Distance and returns the prediction.
%
% INPUTS:
%   q               : query hypervector
%   AM              : Trained associative memory
%
% OUTPUTS:
%   predict_hamm    : prediction 
%

    sims = [];
    am_index = 0;
    
    for j = 1 : classes
        sims(j) = sum(xor(q,aM(am_index)));
        am_index = am_index + 1;
    end
    
    [error, indx]=min(sims');
    M=sort(sims);
    second_error=M(2);
    if indx == 1
        predict_hamm=0;
    else
        predict_hamm=1;
    end
    
    if M(1) == M(2)
        predict_hamm = randi([0,1]); 
    end
    %predict_hamm=indx;
     
end

function [output_vector] = foldedrule90 (D, input_vector, num_folds)
%
% DESCRIPTION computes a random new output_vector using rule90 algorithm given an input_vector
% with dimension D. Does this iteratively over num_folds folds.
% 
% INPUTS:
% D                 : Dimension
% input_vector      : input hv of dim D
% num_folds         : number of sections to do rule90 on
%
% OUTPUTS:
% output_vector     : result of rule90. Also an hv of dim D
%

    fold_width = D / num_folds;
    folded_seed_vector = reshape(input_vector, fold_width, num_folds);
    output_vector = zeros(1,D);
    for f = 1:num_folds
        folded_section = reshape(folded_seed_vector(:,f), 1, fold_width);
        shift_left = circshift(folded_section, -1);
        shift_right = circshift(folded_section, 1);
        xored_shifts = xor(shift_left,shift_right);
        output_vector(:,(f-1)*fold_width+1:f*fold_width) = xored_shifts;
    end
end
