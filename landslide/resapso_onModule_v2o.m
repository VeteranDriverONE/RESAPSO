function [ bestX,bestFitness,Data,NFes,...
    bestFit,bestSolution,Real,Ypred,Ytest,Pred,Pred2,cal_num] =resapso_onModule_v2o(agent_num,lb,ub,fitnessF,func)
% agent_num：种群大小，是一个整数
% lb:下界，是一个D维向量
% ub：上界，是一个D维向量
% func:函数，是一个引用
% 拉丁超立方体采样

Real=[];
Ypred=[];
Ytest=[];
Pred=[];
Pred2=[];
bestSolution=[];
bestFit=Inf;
cal_num=0;

dim=length(lb);
Data=srgtsDOELHS(5*dim,dim,5).*(ub-lb)+lb;
% agent_num=size(Data,1);
Data_fit=zeros(size(Data,1),1);
for i=1:size(Data,1)
    [bestFit,bestSolution,Real,Ypred,Ytest,Pred,Pred2,cal_num,Data_fit(i)]...
        =fitnessF(Data(i,:),func,bestFit,bestSolution,Real,Ypred,Ytest,Pred,Pred2,cal_num);
end
Data=[Data,Data_fit];
P_theta=1/3;
P_x_theta=ones(size(Data,1),4);%第1列表示诞生时的迭代次数，后3列分别表示当前累计的P_x_theta在模型1、2、3上的值
P_x_theta(:,2:4)=1/3;

Vmax=(ub-lb)/10;
Vmin=-Vmax;
% max_iter=500;
max_iter = 1
% sigma=mean(ub-lb)/(20*3);
% neig=ceil((ub-lb)/(11));
np=(ub-lb)./sum(ub-lb);
neig=(ub-lb)./(((5*dim)/prod(np))^(1/dim).^np)/2;
% ga_th=0.1;
struct_times=1;
while size(Data,1)<11*dim
    struct_times=struct_times+1;
    %训练模型
    % kriging
    srgtOPTKRG  = srgtsKRGSetOptions(Data(:,1:dim), Data(:,dim+1));
    srgtSRGTKRG = srgtsKRGFit(srgtOPTKRG);
    % polynomial response surface
    srgtOPTPRS  = srgtsPRSSetOptions(Data(:,1:dim), Data(:,dim+1));
    srgtSRGTRPS = srgtsPRSFit(srgtOPTPRS);
    % radial basis function
    srgtOPTRBF  = srgtsRBFSetOptions(Data(:,1:dim), Data(:,dim+1));
    srgtSRGTRBF = srgtsRBFFit(srgtOPTRBF);
    
%     history=serachAlgorithm(agent_num,Vmax,Vmin,max_iter)
    
    agent=rand(agent_num,dim).*(ub-lb)+lb;
    v=rand(agent_num,dim).*(Vmax-Vmin)+Vmin;    
    
    [agent_fit,f123]=evaluate_fitness(agent,Data,P_theta,P_x_theta,srgtSRGTKRG,srgtSRGTRPS,srgtSRGTRBF,neig);
    
    [~,min_fit_pos]=min(agent_fit);
    agent=[agent,agent_fit];
    g_best=agent(min_fit_pos,:);
    agent_history=agent;
    
    history=agent;
    history_f123=f123;
    
    t=1;
    c1=2;
    c2=2;
    while t<max_iter  %在代理模型上用PSO搜索
        w=1.2-0.8*t/max_iter;
        v=v*w+c1*rand(agent_num,1).*(g_best(1:dim)-agent(:,1:dim))+c2*rand(agent_num,1).*(agent_history(:,1:dim)-agent(:,1:dim));
        v=bounds(v,Vmin,Vmax);
        agent(:,1:dim)=agent(:,1:dim)+v;
        agent(:,1:dim)=bounds(agent(:,1:dim),lb,ub);

        [agent_fit,f123]=evaluate_fitness(agent(:,1:dim),Data,P_theta,P_x_theta,srgtSRGTKRG,srgtSRGTRPS,srgtSRGTRBF,neig);
        agent(:,dim+1)=agent_fit;

        [min_fit,min_fit_pos]=min(agent(:,dim+1));
        if min_fit<g_best(dim+1)
            g_best=agent(min_fit_pos,:);
        end
        agent_history(agent(:,dim+1)<agent_history(:,dim+1),:)=agent(agent(:,dim+1)<agent_history(:,dim+1),:);

        t=t+1;
        history=[history;agent];
        history_f123=[history_f123;f123];
    end      
    
    %直觉模糊多属性决策，选出最不确定和最优个体
    [index_best,index_worst]=IFS_decision(history,history_f123,Data,neig);
    
    %计算正反馈
    if min(sum(abs(history(index_best,1:dim)-Data(:,1:dim)),2))>0
        [bestFit,bestSolution,Real,Ypred,Ytest,Pred,Pred2,cal_num,real_fit]=fitnessF(...
            history(index_best,1:dim),func,bestFit,bestSolution,Real,Ypred,Ytest,Pred,Pred2,cal_num);
        [temp_f1,~] = srgtsKRGPredictor(history(index_best,1:dim), srgtSRGTKRG);
        [temp_f2,~] = srgtsPRSPredictor(history(index_best,1:dim), Data(:,1:dim), srgtSRGTRPS);
        temp_f3 = srgtsRBFEvaluate(history(index_best,1:dim), srgtSRGTRBF);
        temp_x=abs(real_fit-[temp_f1,temp_f2,temp_f3]);
        temp_x=temp_x+1;
        theta_t=(1./temp_x)./sum(1./temp_x,2);
        
        domain=prod(abs(history(index_best,1:dim)-Data(:,1:dim))<neig,2);
        P_iter=P_x_theta(domain==1,1);
        P_x_theta(domain==1,2:4)=P_x_theta(domain==1,2:4).*P_iter./(P_iter+1)+theta_t./(P_iter+1);
        P_x_theta(domain==1,1)=P_iter+1;
        
        Data=[Data;[history(index_best,1:dim),real_fit]];
        P_x_theta=[P_x_theta;[1,theta_t]];
    end
    
    if min(sum(abs(history(index_worst,1:dim)-Data(:,1:dim)),2))>0
        [bestFit,bestSolution,Real,Ypred,Ytest,Pred,Pred2,cal_num,real_fit]=fitnessF(...
            history(index_worst,1:dim),func,bestFit,bestSolution,Real,Ypred,Ytest,Pred,Pred2,cal_num);
        [temp_f1,~] = srgtsKRGPredictor(history(index_worst,1:dim), srgtSRGTKRG);
        [temp_f2,~] = srgtsPRSPredictor(history(index_worst,1:dim), Data(:,1:dim), srgtSRGTRPS);
        temp_f3 = srgtsRBFEvaluate(history(index_worst,1:dim), srgtSRGTRBF);
        temp_x=abs(real_fit-[temp_f1,temp_f2,temp_f3]);
        temp_x=temp_x+1;
        theta_t=(1./temp_x)./sum(1./temp_x,2);
        
        domain=prod(abs(history(index_worst,1:dim)-Data(:,1:dim))<neig,2);
        P_iter=P_x_theta(domain==1,1);
        P_x_theta(domain==1,2:4)=P_x_theta(domain==1,2:4).*P_iter./(P_iter+1)+theta_t./(P_iter+1);
        P_x_theta(domain==1,1)=P_iter+1;
        
        Data=[Data;[history(index_worst,1:dim),real_fit]];
        P_x_theta=[P_x_theta;[1,theta_t]];
    end
    
end
[bestFitness,best_pos]=min(Data(:,end));
bestX=Data(best_pos,:);
NFes=zeros(1,11*dim);
for i=1:11*dim
    NFes(i)=min(Data(1:i,dim+1));
end
% figure
% plot(Data(:,1),Data(:,2),'b*');
% hold on
% for i=1:size(Data,1)
% %     radius=2*sigma^2*log(1/ga_th);
%     rectangle('Curvature', [0 0],'Position', [Data(i,1:2)-neig(1:2),neig(1:2)*2], 'edgecolor', 'r','facecolor','none')
% end
end
%%
% 边界函数
function [X]=bounds(X,lb,ub)
    X=(X>ub).*ub+(X<=ub).*X;
    X=(X<lb).*lb+(X>=lb).*X;
end
% 在代理模型上评估适应度
function [fit,f123]=evaluate_fitness(agent,Data,P_theta,P_x_theta,srgtSRGTKRG,srgtSRGTRPS,srgtSRGTRBF,neig)
    [agent_num,dim]=size(agent);
    fit=zeros(agent_num,1);
    [f1,~] = srgtsKRGPredictor(agent(:,1:dim), srgtSRGTKRG);
    [f2,~] = srgtsPRSPredictor(agent(:,1:dim), Data(:,1:dim), srgtSRGTRPS);
    f3 = srgtsRBFEvaluate(agent(:,1:dim), srgtSRGTRBF);
    f123=[f1,f2,f3];
    for i=1:agent_num
        domain=prod(abs(agent(i,:)-Data(:,1:dim))<neig,2);
        screen_dist=sqrt(sum((agent(i,:)-Data(domain==1,1:dim)).^2,2));
        screen_dist=screen_dist+1;  %防止分母为0
        wight=(1./screen_dist)/sum(1./screen_dist); %计算权重
        if size(wight,1)~=0
            P_X=sum(P_x_theta(domain==1,2:4).*P_theta,2);
            accu=sum(P_x_theta(domain==1,2:4).*P_theta.*f123(i,:)./P_X,2);
            fit(i)=sum(wight.*accu);
        else
            fit(i)=mean(f123(i,:));
        end
    end
end
%直觉模糊多属性决策
function [index_prominence,index_uncertainty]=IFS_decision(agent,f123,Data,neig)
    index_prominence=IFS_decision_pro(agent(:,1:end-1),agent(:,end),f123,Data,neig); %不需要计算适应度
    index_uncertainty=IFS_decision_unc(agent(:,1:end-1),f123,Data,neig);
end
%直觉模糊多属性决策选择最优点
function [index_prominence]=IFS_decision_pro(agent,fit,f123,Data,neig)
    [agent_num,dim]=size(agent);
    minF=min(min(f123));
    x1=zeros(agent_num,3);  %1-pi-gamma/max/min
%     loged_fit=log(f123-minF+exp(1));
%     maxF=max(max(loged_fit));
%     minF=min(min(loged_fit));
%     x1(:,2)=0.5*(max(loged_fit,[],2)-minF)/(maxF-minF);
%     x1(:,3)=0.5*(min(loged_fit,[],2)-minF)/(maxF-minF);
%     x1(:,1)=1-x1(:,2)-x1(:,3);
    loged_fit123=log(f123-minF+exp(1));
    loged_fit=log(fit-min(fit)+exp(1));
    mean_fit=mean(loged_fit123,2);
    x1(:,2)=0.5*mean_fit/max(mean_fit);
    x1(:,3)=0.5*(loged_fit-min(loged_fit))/(max(loged_fit)-min(loged_fit));
    x1(:,1)=1-x1(:,2)-x1(:,3);
    
    x2=zeros(agent_num,3); %邻域内其它Data数/1-u-pi/与Data的最近距离
    for i=1:agent_num
        x2(i,3)=min(sqrt(sum((agent(i,:)-Data(:,1:dim)).^2,2)));
        x2(i,1)=sum(prod(abs(agent(i,:)-Data(:,1:dim))<neig,2));
%         gauss_dist=exp(-sqrt(sum((agent(i,:)-Data(:,1:dim)).^2,2))/(2*sigma^2));
%         x2(i,1)=sum(gauss_dist>gauss_thre);
    end
    x2(:,3)=0.5*x2(:,3)/max(x2(:,3));
    x2(:,1)=0.5*x2(:,1)/max(max(x2(:,1)),1);
    x2(:,2)=1-x2(:,1)-x2(:,3);
    index_prominence=decision_process(x1,x2);
end
%直觉模糊多属性决策选择最不确定点
function [index_uncertainty]=IFS_decision_unc(agent,f123,Data,neig)
    [agent_num,dim]=size(agent);    
    maxF=max(max(f123));
    minF=min(min(f123));
    x1=zeros(agent_num,3);  %max-min/量纲/1-u-pi
    diff=max(f123,[],2)-min(f123,[],2);
    x1(:,1)=0.5*diff/(maxF-minF);
    x1(:,2)=0.5*log10(diff)/max(log10(diff));
    x1(:,3)=1-x1(:,1)-x1(:,2);
    
    x2=zeros(agent_num,3); %与Data的最近距离/1-u-ga/邻域内其它粒子数
    for i=1:agent_num
        x2(i,1)=min(sqrt(sum((agent(i,:)-Data(:,1:dim)).^2,2)));
%         gauss_dist=exp(-sqrt(sum((agent(i,:)-Data(:,1:dim)).^2,2))/(2*sigma^2));
%         x2(i,3)=sum(gauss_dist>gauss_thre);
        x2(i,3)=sum(prod(abs(agent(i,:)-Data(:,1:dim))<neig,2));
    end
    x2(:,1)=0.5*x2(:,1)/max(x2(:,1));
    x2(:,3)=0.5*x2(:,3)/max(max(x2(:,3)),1);
    x2(:,2)=1-x2(:,1)-x2(:,3);
    index_uncertainty=decision_process(x1,x2);
end
% 决策过程
function index=decision_process(a1,a2)
    w=[1,0,0;
        1,0,0];  %属性一和属性二的直觉模糊权重
    w_Fin=(w(:,1)+w(:,2)/2)/sum(w(:,1)+w(:,2)/2); %对称权系数合成权重
    A=a1*w_Fin(1)+a2*w_Fin(2);
    G=[1,0,0];  %理想解
    B=[0,0,1];  %非理想解
    dist_G=Dist(A,G);
    dist_B=Dist(A,B);
    xi=dist_B./(dist_G+dist_B);
    [~,index]=max(xi);
end
%海明距计算决策结果
function [dist]=Dist(X,Y) 
    dist=sum(abs(X-Y),2)/2;
end
%计算适应度
function [bestFit,bestSolution,Real,Ypred,Ytest,Pred,Pred2,cal_num,Y]...
    =fitnessF(x,func,bestFit,bestSolution,Real,Ypred,Ytest,Pred,Pred2,cal_num)
    cal_num=cal_num+1;
    x=round(x);
    disp(x);
    data=func.landslide(double(x));
    real1 = double(py.array.array('d',py.numpy.nditer(data(1))));
    y_pred = double(py.array.array('d',py.numpy.nditer(data(2))));
    y_test = double(py.array.array('d',py.numpy.nditer(data(3))));
    pred = double(py.array.array('d',py.numpy.nditer(data(4))));
    pred2 = double(py.array.array('d',py.numpy.nditer(data(5))));
    Y=sum(abs(y_pred-y_test))*100;
    if Y<bestFit
        bestFit=Y;
        bestSolution=x;
        Real=real1;
        Ypred=y_pred;
        Ytest=y_test;
        Pred=pred;
        Pred2=pred2;
    end
end