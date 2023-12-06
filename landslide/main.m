% clear;
% clc;

% define search space and fitness function
fitnessF = @Sphere;
maxIter=1; 
pop=100;
% search space (refer our paper)
lb=[0.6,0.6,1.5,3.5,-0.4]; 
ub=[32.4,32.4,7.4,48.4,2.4];
dim=length(lb);

% load python model as fitness function
func=py.importlib.import_module('landslide_main');
py.importlib.reload(func);

% static value
global bestSolution;
global bestFit;
global Real;
global Ypred;
global Ytest;
global Pred;
global Pred2;
bestFit=Inf;

% run optimizer
run_num = 20;
com_bestSolution = zeros(run_num,dim);
for i=1:run_num
    [bestX, bestFitness, cur, goal] = resapso_onModule_v2o(pop,lb,ub,fitnessF,func);
    com_bestSolution(i,:) = bestX;
end

%%
% static avrage value
test_num = 20;
avg_fit=zeros(length(com_bestSolution),test_num);
for j=1:length(com_bestSolution)
    disp(com_bestSolution(j,:));
    for k=1:test_num
        data = land_main.landslide(double(round(com_bestSolution(j,:))));
        y_pred = double(py.array.array('d',py.numpy.nditer(data(2))));
        y_test = double(py.array.array('d',py.numpy.nditer(data(3))));
        avg_fit(j,k) = sum(abs(y_pred-y_test))*100;
    end
end
save('algs.mat','avg_fit');
clearvars -except algs i test_num land_main

%%

function [y] = Sphere(x, func)
    % calculate fitness of landslie
    global bestSolution;
    global bestFit;
    global Real;
    global Ypred;
    global Ytest;
    global Pred;
    global Pred2;
    
    x=round(x);
    disp(x);
    data=func.landslide(double(x));
    real = double(py.array.array('d',py.numpy.nditer(data(1))));
    y_pred = double(py.array.array('d',py.numpy.nditer(data(2))));
    y_test = double(py.array.array('d',py.numpy.nditer(data(3))));
    pred = double(py.array.array('d',py.numpy.nditer(data(4))));
    pred2 = double(py.array.array('d',py.numpy.nditer(data(5))));
    y=sum(abs(y_pred-y_test))*100;
    if y < bestFit
        bestFit=y;
        bestSolution=x;
        Real=real;
        Ypred=y_pred;
        Ytest=y_test;
        Pred=pred;
        Pred2=pred2;
    end
end