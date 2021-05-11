plotting = zeros(1,50);
for x = 1:1:50
    plotting(1,x) = vector_counter(x);
end
plot(1:1:50,plotting,'LineWidth',4);
title('Unique feature channel sets generated vs. vectors stored')
ylabel('Unique feature channel sets');
xlabel('vectors stored');



function num_vectors = vector_counter(x)
num_vectors = 0;    
subtracter = 1;
    while (subtracter < (x-1))
        num_vectors = num_vectors + floor((x-subtracter)/2);
        subtracter = subtracter + 1;
    
    end
end


