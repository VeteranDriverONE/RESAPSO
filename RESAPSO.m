function [ bestX,bestFitness,Data,NFes] =RESAPSO(lb, ub, func, func_id)
if ~exist('func','var')
    disp('---------Initialization parameters----------')
    agent_num=100;
    ub=ones(1,10)*100;
    lb=-ub;
    func_id=1;
    func=str2func('cec17_func');
end
% agent_num: Size of population
% lb: Lower bound, a D-dimensional vector
% ub：Upper bound, a D-dimensional vector
% func: Objective function


dim=length(lb);
Data=srgtsDOELHS(5*dim,dim,5).*(ub-lb)+lb; % Latin hypercube sampling
Data_fit=func(Data',func_id);
Data=[Data,Data_fit'];
P_theta=1/3;  % The priori reliability of each surrogate model
P_x_theta=ones(size(Data,1),4); 
% an n*4 matrix, with n representing the number of evaluation points (which increases as the algorithm runs); 
% the first column represents the number of times the evaluation point predicts the selected individual, 
% and 2-4 represent the three surrogate model likelihood values, respectively.
P_x_theta(:,2:4)=1/3;  % Initially, each surrogate model frequency is constant

Vmax=(ub-lb)/10;  % The maximum velocity of a particle
Vmin=-Vmax;  % The minimum velocity of a particle
max_iter=500;  % The number of population iterations (evaluated on the surrogate model)
np=(ub-lb)./sum(ub-lb);
neig=(ub-lb)./(((5*dim)/prod(np))^(1/dim).^np)/2;  % The neighborhood
struct_times=1;
while size(Data,1)<11*dim
    struct_times=struct_times+1;
    % Training the model
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
    
    % ------------ PSO initialization ------------ %
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
    % Search with PSO on the surrogate model
    while t<max_iter  
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
    
    % Intuitive fuzzy multi-attribute decision making, selecting the
    % uncertain individual and the promising individual
    [index_best,index_worst]=IFS_decision(history,history_f123,Data,neig);
    
    % Error calculation based on promising point
    if min(sum(abs(history(index_best,1:dim)-Data(:,1:dim)),2))>0
         real_fit=func(history(index_best,1:dim)',func_id)';
        [temp_f1,~] = srgtsKRGPredictor(history(index_best,1:dim), srgtSRGTKRG);
        [temp_f2,~] = srgtsPRSPredictor(history(index_best,1:dim), Data(:,1:dim), srgtSRGTRPS);
        temp_f3 = srgtsRBFEvaluate(history(index_best,1:dim), srgtSRGTRBF);
        temp_x=abs(real_fit-[temp_f1,temp_f2,temp_f3]);
        temp_x=temp_x+1;
        theta_t=(1./temp_x)./sum(1./temp_x,2);
        
        domain=prod(abs(history(index_best,1:dim)-Data(:,1:dim))<neig,2);
        P_iter=P_x_theta(domain==1,1);
%         P_x_theta(domain==1,2:4)=P_x_theta(domain==1,2:4).*P_iter./(P_iter+1)+theta_t./(P_iter+1);
        P_x_theta(domain==1,2:4)=P_x_theta(domain==1,2:4).*theta_t;
        P_x_theta(domain==1,1)=P_iter+1;
        
        Data=[Data;[history(index_best,1:dim),real_fit]];
        P_x_theta=[P_x_theta;[1,theta_t]];
    end
    
    % Error calculation based on uncertain point
    if min(sum(abs(history(index_worst,1:dim)-Data(:,1:dim)),2))>0
        real_fit=func(history(index_worst,1:dim)',func_id)';
        [temp_f1,~] = srgtsKRGPredictor(history(index_worst,1:dim), srgtSRGTKRG);
        [temp_f2,~] = srgtsPRSPredictor(history(index_worst,1:dim), Data(:,1:dim), srgtSRGTRPS);
        temp_f3 = srgtsRBFEvaluate(history(index_worst,1:dim), srgtSRGTRBF);
        temp_x=abs(real_fit-[temp_f1,temp_f2,temp_f3]);
        temp_x=temp_x+1;
        theta_t=(1./temp_x)./sum(1./temp_x,2);
        
        domain=prod(abs(history(index_worst,1:dim)-Data(:,1:dim))<neig,2);
        P_iter=P_x_theta(domain==1,1);
%         P_x_theta(domain==1,2:4)=P_x_theta(domain==1,2:4).*P_iter./(P_iter+1)+theta_t./(P_iter+1);
        P_x_theta(domain==1,2:4)=P_x_theta(domain==1,2:4).*theta_t;
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

end
%%
% Boundary
function [X]=bounds(X,lb,ub)
    X=(X>ub).*ub+(X<=ub).*X;
    X=(X<lb).*lb+(X>=lb).*X;
end
% Evaluating fitness on an surrogate model(BES) - weight 
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
        weight=(1./screen_dist)/sum(1./screen_dist); %计算权重
        if size(weight,1)~=0
            P_X=sum(P_x_theta(domain==1,2:4).*P_theta,2);
            accu=sum(P_x_theta(domain==1,2:4).*P_theta.*f123(i,:)./P_X,2);
            fit(i)=sum(weight.*accu);
        else
            fit(i)=mean(f123(i,:));
        end
    end
end
% Evaluating fitness on an surrogate model(BES) - max
function [fit,f123]=evaluate_fitness2(agent,Data,P_theta,P_x_theta,srgtSRGTKRG,srgtSRGTRPS,srgtSRGTRBF,neig)
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
        weight=(1./screen_dist)/sum(1./screen_dist); %计算权重
        if size(weight,1)~=0
            P_X=sum(P_x_theta(domain==1,2:4).*P_theta,2);
            P_theta_x = P_x_theta(domain==1,2:4).*P_theta./P_X;
            [~, P_index] = max(P_theta_x,[],2);
            [~, w_index] = max(weight);
            fit(i) = f123(i,P_index(w_index));
        else
            fit(i)=mean(f123(i,:));
        end
    end
end

% Intuitionistic fuzzy multi-attribute decision making (IFMADA)
function [index_prominence,index_uncertainty]=IFS_decision(agent,f123,Data,neig)
    index_prominence=IFS_decision_pro(agent(:,1:end-1),agent(:,end),f123,Data,neig);
    index_uncertainty=IFS_decision_unc(agent(:,1:end-1),f123,Data,neig);
end

% Select the promising point with IFMADA
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
    
    x2=zeros(agent_num,3);
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

% Select the uncertain point with IFMADA
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

% Decision-making process
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

% Hemming distance calculation decision results
function [dist]=Dist(X,Y) 
    dist=sum(abs(X-Y),2)/2;
end