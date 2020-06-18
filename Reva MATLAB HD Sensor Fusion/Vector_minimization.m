num_vectors = 0;
x = 0;

%counts number of vectors needed for at least 105 combinations
%output is 23 necessary vectors for 110 combinations
while (num_vectors < 105)
    num_vectors = vector_counter(x);
    x = x + 1;
end

%list of 23 sample vectors
vectors = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W'];

%110 output combinations given 23 sample vectors
combinations = final_arrange(vectors);
combinations

%uses arrange_vectors on a list of vectors
function complete_array = final_arrange(m)
complete_array = [];
    while (length(m)>2)
        complete_array = [complete_array; arrange_vectors(m)];
        m(1) = [];
    end
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

%given a number of vectors, counts all combinations
function num_vectors = vector_counter(x)
num_vectors = 0;    
subtracter = 1;
    while (subtracter < (x-1))
        num_vectors = num_vectors + floor((x-subtracter)/2);
        subtracter = subtracter + 1;
    
    end
end

