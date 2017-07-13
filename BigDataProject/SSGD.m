function [corrections, new_w, mistakes, accuracy,negative_ll] = SSGD(X,labels,X_test,labels_test, epochs, sigma,r0)
 
C= sigma*sigma;
no_of_rows = size(X_test,1);
%initialize weight vector w = col vector of size size(X,2) 
W = zeros(size(X,2),1);

rt = r0; %initial learning rate
X = X';
corrections = 0;


for i = 1 : epochs  
    
      additiveterm = 0;
    
      rt = rt/(1+((rt*i)/C)); 
      % code to shuffle
      X = [X', labels]; %combine into 1 matrix
      X = X(randperm(size(X,1)), :); %randomly permute indices
      labels = X(:,size(X,2)); %strip off labels
      X = X(:,1:size(X,2)-1); %strip off X vector
      X = X';
      
  %Send
  for j = 1 : size(X,2) 
      
    %Uncomment this for SVM:
%      if labels(j)*W'*X(:,j) <= 1 %wrong
%       corrections = corrections + 1;
%       W = (1-rt)*W + rt*C*labels(j)*X(:,j);  %update W
%      
%     else
%       W = (1-rt)*W;
%     end
     %Uncomment this for Logistic Regression 
     W = (1-(rt*2/(sigma*sigma)))*W + ((rt*labels(j)*X(:,j)*(exp(-labels(j)*W'*X(:,j))))/(1+exp(-labels(j)*W'*X(:,j))));  %update W Note: C = sigma^2
    additiveterm = additiveterm + log(1+exp(-labels(j)*W'*X(:,j)));
 
  end
  negative_ll(i) = -additiveterm;
end

new_w = W;
mistakes = sum(sign(new_w'*X_test')~=labels_test');

accuracy = ((no_of_rows-mistakes)/no_of_rows) * 100;