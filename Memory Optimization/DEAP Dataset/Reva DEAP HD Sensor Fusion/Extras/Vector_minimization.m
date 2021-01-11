outputs = 0;
num_vectors = 0;

%counts number of vectors needed for at least 105 combinations
%output is 23 necessary vectors for 110 combinations
while (outputs < 32)
    outputs = vector_counter(num_vectors);
    num_vectors = num_vectors + 1;
end
num_vectors

m = zeros(2);
m(3,4) = 4;


%list of 23 sample vectors
vectors = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W'];

%110 output combinations given 23 sample vectors
combinations = final_arrange(vectors);

x = zeros(5);
x = [x;x];

D = 2000;
acc1 = 12;
acc2 = 1999;
randCounter = 10;
while (randCounter>1)
    acc1 = acc1 + 1;
    acc2 = acc2 - 1;
    randCounter = randCounter-1;
    randCounter
    x((randCounter*2-1),(D/1000)) = acc1;
    x((randCounter*2),D/1000)= acc2;

end



%given the first vector, creates all possible vector combinations
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

