function [label, iter_num, center, obj] = kmeans_with_new_formulation(X, k, label)
% INPUT:
% X: dim*n matrix, each column is a data point

n = size(X,2);
%Initialization of F
if nargin < 3
label = ceil(k*rand(n,1));  % random initialization
end
F = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
F = full(F);

tic
A = X'*X;
s = ones(k,1);
M = ones(n,k);

for iter = 1:1000         
    label_last = label;  
    for i = 1:k
        f = F(:,i); 
        temp1 = f'*A*f;
        temp2 = f'*f;
        s_i = sqrt(temp1)/temp2;
        s(i) = s_i;       
    end  
    for iter_re = 1:100
       for j = 1:k
           f = F(:,j);
           temp4 = A*f;
           temp3 = sqrt(f'*temp4);
           m_j = (1/temp3)*temp4;
           M(:,j) = m_j;
       end
       temp_s = s';
       S = repmat(temp_s,n,1);
       temp_M = 2*S.*M;
       temp_S = S.^2;
       temp5 = temp_S - temp_M;
       [~,label_new] = min(temp5,[],2);

       F = sparse(1:n,label_new,1,n,k,n);  % transform label into indicator matrix
       F = full(F);
                           
       if(label == label_new)
           break;
       end
       label = label_new;
    end
    
    if(label == label_last)
        iter_num = iter;
        break;
    end  
end
time=toc;
center = X*(F*spdiags(1./sum(F,1)',0,k,k));    % compute center of each cluster
temp6 = X-center*F';
obj = norm(temp6,'fro');
obj = obj^2;
