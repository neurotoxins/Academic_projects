function [all_accuracy] = BigDataLogisticRegression(train_X,train_labels,test_X,test_labels)
    epochs = 10;
    
    no_of_rows = size(train_X,1);
    for i = 1:10
        chunk(i) = round(i* (no_of_rows/10));
    end
    sigma = 10;
    r_0 = 10;

    counter = 1;
    for i = 1:10
        X_test = test_X;
        labels_test = test_labels;
        X = train_X(1:chunk(i),:); 
        labels = train_labels(1:chunk(i),:);
        [corrections,weight,mistakes,accuracy,negative_ll] = SSGD(X,labels,X_test,labels_test,epochs,sigma,r_0);
        all_accuracy(counter) = accuracy;
        counter = counter +1;
    end

end