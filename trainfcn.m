%%%%%%%%%%
%author: Gokul Subraveti

%Note: For each initial column profile, unique step simulations are
%carried out using the detailed model to obtain spatiotemporal solutions of four state variables. 
%In this example, 60 different detailed model simulations are performed in total.
%This code processes the data from simulations to form as appropriate inputs
%to neural network training code. 

%%% train_data.mat consists of data structures
%y_s = structure containing CO2 gas-phase composition spatiotemporal
%solutions of 60 different simulation cases 
%P_s = structure containing non-dimensionalized column pressure spatiotemporal
%solutions of 60 different simulation cases 
%qa_s = structure containing non-dimensionalized CO2 solid-phase concentration 
%spatiotemporal solutions of 60 different simulation cases 
%qb_s = structure containing non-dimensionalized N2 solid-phase concentration 
%spatiotemporal solutions of 60 different simulation cases 
%t = temporal data points 
%z = spatial locations across the column
%coeff = PDE coefficients 
%cin = matrix containing 60 different initial profiles

%%%%%%%%%%

%
load cin
load train_data
nsamples=60; %number of initial column profiles N_k
%

%for loop below converts data structures to array
for i=1:nsamples
Exacty(:,:,i)=transpose(y_s.(sprintf('dat%d', i))); 
Exactp(:,:,i)=transpose(P_s.(sprintf('dat%d', i)));
Exactqa(:,:,i)=transpose(qa_s.(sprintf('dat%d', i)));
Exactqb(:,:,i)=transpose(qb_s.(sprintf('dat%d', i)));
end

[Z, T] = meshgrid(z,t); % spatiotemporal 2D grid
% 

%indices where the final column pressure (non-dimensionalized) in the
%blowdown step is 0.7 (lower limit in the present work)
for i=1:nsamples
idx = find(P_s.(sprintf('dat%d', i))(1,:)<0.7);
b(i)=idx(1)-1;
end

X_sol_1=[Z(:) T(:) cin(1,:).*ones(length(Z(:)),1)]; %concatenated input matrix to 
%neural network corresponding to simulation case#1 from N_k=60

y_sol_1=Exacty(:,:,1); y_sol_1=y_sol_1(:); %create an output vector representing original spatiotemporal solutions of y corresponding to X_sol_1 
p_sol_1=Exactp(:,:,1); p_sol_1=p_sol_1(:); %create an output vector representing original spatiotemporal solutions of P corresponding to X_sol_1 
qa_sol_1=Exactqa(:,:,1); qa_sol_1=qa_sol_1(:); %create an output vector representing original spatiotemporal solutions of qCO2 corresponding to X_sol_1 
qb_sol_1=Exactqb(:,:,1); qb_sol_1=qb_sol_1(:); %create an output vector representing original spatiotemporal solutions of qN2 corresponding to X_sol_1 

low_bound=min(X_sol_1(:,1:2)); %lower bounds of z and t
up_bound=max(X_sol_1(:,1:2)); %upper bounds of z and t

C0_low_bound=min(cin); %lower bounds of y0(z), i,e., initial gas-phase composition profile
C0_up_bound=max(cin); %upper bounds of y0(z), i,e., initial gas-phase composition profile

N_c0=250; %number of collocation points for each index #j

%create concatenated input X matrices representing the spatiotemporal domain at initial,
%boundary, final, and collocation points
X0=[]; %concetenated X matrix representing the initial state
X_en=[]; %concatenated X matrix representing the final state
X_lb=[]; %concatenated X matrix representing the left boundary
X_rb=[]; %concatenated X matrix representing the right boundary
X_c_train=[]; %concatenated X matrix representing collocation points

in_train=[]; %concatenated output matrix with four state variables containing the initial data
en_train=[]; %concatenated output matrix with four state variables containing the final data
lb_train=[]; %concatenated output matrix with four state variables containing the left boundary data
rb_train=[]; %concatenated output matrix with four state variables containing the right boundary data

%for loop below stacks initial and boundary labelled data from all
%simulations and also concetenates X_c_train matrix of collocation points 
for j=1:nsamples
    
    xx1=[Z(1,:)' T(1,:)' cin(j,:).*ones(length(Z(1,:)),1)]; %input matrix corresponding to initial condition
    xx2=[Z(b(j),:)' T(b(j),:)' cin(j,:).*ones(length(Z(end,:)),1)]; %input matrix corresponding to final state
    xx3=[Z(:,1) T(:,1) cin(j,:).*ones(length(Z(:,1)),1)]; %input matrix corresponding to left boundary
    xx4=[Z(:,end) T(:,end) cin(j,:).*ones(length(Z(:,end)),1)]; %input matrix corresponding to right boundary
    
    %introducing stochasticity to pick 300 random boundary data points from a set of 1001
    N_u=randperm(size(xx3,1)); xx3=xx3(N_u(1:300),:);xx4=xx4(N_u(1:300),:);
    %
    yy1=transpose(Exacty(1,:,j)); %initial CO2 gas-phase composition 
    pp1=transpose(Exactp(1,:,j)); %initial column pressure
    qa1=transpose(Exactqa(1,:,j)); %initial CO2 solid-phase concentration
    qb1=transpose(Exactqb(1,:,j)); %initial N2 solid-phase concentration
    
    in0_train=[yy1 pp1 qa1 qb1]; %concatenated initial state labelled data
    %
    yy2=transpose(Exacty(b(j),:,j)); %final CO2 gas-phase composition
    pp2=transpose(Exactp(b(j),:,j)); %final column pressure
    qa2=transpose(Exactqa(b(j),:,j)); %final CO2 solid-phase concentration
    qb2=transpose(Exactqb(b(j),:,j)); %final N2 solid-phase concentration
    
    en0_train=[yy2 pp2 qa2 qb2]; %concatenated final state labelled data
    % 
    yy3=Exacty(:,1,j); %left boundary CO2 gas-phase composition (at z=0)
    pp3=Exactp(:,1,j); %left boundary column pressure (at z=0)
    qa3=Exactqa(:,1,j); %left boundary CO2 solid-phase concentration (at z=0)
    qb3=Exactqb(:,1,j); %left boundary N2 solid-phase concentration (at z=0)
    
    lb0_train=[yy3 pp3 qa3 qb3]; lb0_train=lb0_train(N_u(1:300),:); %concatenated left boundary labelled data
    %
    yy4=Exacty(:,end,j); %right boundary CO2 gas-phase composition (at z=0)
    pp4=Exactp(:,end,j); %right boundary column pressure (at z=0)
    qa4=Exactqa(:,end,j); %right boundary CO2 solid-phase concentration (at z=0)
    qb4=Exactqb(:,end,j); %right boundary N2 solid-phase concentration (at z=0)
    
    rb0_train=[yy4 pp4 qa4 qb4]; rb0_train=rb0_train(N_u(1:300),:); %concatenated right boundary labelled data

    %
    x0_train=[xx1;xx3;xx4]; %input matrix corresponding to initial and boundary region
    x0_c_train=low_bound + (up_bound-low_bound).*lhsdesign(N_c0,2); %generate random collocation points
    x0_c_train=[x0_c_train cin(j,:).*ones(length(x0_c_train),1)]; %create concatenated X matrix for collocation points
    x0_c_train=[x0_c_train;x0_train]; %adding initial and boundary points to collocation matrix
    
    X0=[X0;xx1]; %stack initial X matrix from all simulations
    X_en=[X_en;xx2]; %stack final X matrix from all simulations
    X_lb=[X_lb;xx3]; %stack left boundary X matrix from all simulations
    X_rb=[X_rb;xx4]; %stack right boundary X matrix from all simulations
    X_c_train=[X_c_train;x0_c_train]; %stack collocation X matrix for all cases
    
    in_train=[in_train;in0_train]; %stack initial data from all simulations
    en_train=[en_train;en0_train]; %stack final data from all simulations
    lb_train=[lb_train;lb0_train]; %stack left boundary data from all simulations
    rb_train=[rb_train;rb0_train]; %stack right boundary data from all simulations
    
  

end

%create index IDs for the loss function in neural network training code
N0_ids=zeros(size(in0_train,1),1);
for j=2:nsamples
N0_ids=[N0_ids; (j-1).*ones(size(in0_train,1),1)]; 
end

N_b_ids=zeros(size(lb0_train,1),1);
for j=2:nsamples
N_b_ids=[N_b_ids; (j-1).*ones(size(lb0_train,1),1)]; 
end

N_c_ids=zeros(size(x0_c_train,1),1);
for j=2:nsamples
N_c_ids=[N_c_ids; (j-1).*ones(size(x0_c_train,1),1)]; 
end

low_bound=[low_bound C0_low_bound];
up_bound=[up_bound C0_up_bound];

save train_ads.mat Z T X0 X_en X_lb X_rb X_c_train in_train en_train lb_train rb_train...
    X_sol_1 y_sol_1 p_sol_1 qa_sol_1 qb_sol_1 N0_ids N_b_ids N_c_ids up_bound low_bound coeff
   

clear