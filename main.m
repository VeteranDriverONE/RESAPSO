clear
rng('shuffle');
func=@cec17_func;

dim=10;
func_num=30;
test_num=20;
ub=ones(1,dim)*100;
lb=-ub;
agent_num=100;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%RESAPSO
BestXs_RESAPSO=cell(func_num,1);
BestFitnesses_RESAPSO=zeros(func_num,test_num);
NFes_RESAPSO=zeros(func_num,dim*11);
for func_id=1:func_num
    temp_bestX=zeros(test_num,dim);
    temp_fitness=zeros(1,test_num);
    temp_nfes=zeros(1,dim*11);
    parfor j=1:test_num
        [bestX,bestFitness,~,NFes]=RESAPSO(lb,ub,func,func_id);
        temp_bestX(j,:)=bestX(1:dim);
        temp_fitness(1,j)=bestFitness;
        temp_nfes=temp_nfes+NFes;
    end
    BestXs_RESAPSO{func_id,1}=temp_bestX;
    BestFitnesses_RESAPSO(func_id,:)=temp_fitness;
    NFes_RESAPSO(func_id,:)=temp_nfes/test_num;
end
static_RESAPSO=[mean(BestFitnesses_RESAPSO,2),sum((BestFitnesses_RESAPSO-mean(BestFitnesses_RESAPSO,2)).^2,2)/test_num];
save(['result/temp/static_RESAPSO.mat'],"static_RESAPSO");
save(['result/temp/BestFitnesses_RESAPSO.mat'],"BestFitnesses_RESAPSO");
save(['result/temp/NFes_RESAPSO.mat'],"NFes_RESAPSO");
save(['result/temp/BestXs_RESAPSO.mat'],"BestXs_RESAPSO");

