%% Two side IC with public goods (sharing rule as state variable)
% February 2021
% State variables: sharing rule, agent 2's promised utility. 
% this code solves the infinite horizon two-side IC problem with two agents
% and private consumption and public good consumption. The outer loop converges on the possible
% utility gridpoints for agent 2 (see Wang 1995), the inner loop converges
% on P function that maps utility of agent 2 to utility of agent 1.
% Within the inner loop, maximization is done using linprog with a
% probability vector (see SL exercise 20.4 and Karaivanov's Matlab code for
% moral hazard).

%function [w0, w_feasible, P, nfeas, X_all] = ic_sharingrule(a1,a2,y1min,y1max,y2min,y2max,alpha,beta,price,n,ns,nw)

clear all; close all; clc;

%% Parameters set-up
%
% discounting factor
beta = 0.94;
% preference of private consumption
a1 = 0.5;
a2 = 0.5;
% price of public good
price = 1;
% number of gridpoints
n = 2;
ns = 25;
nw = 40;
% income processes
y1min = 1;
y1max = 5;
y2min = 1;
y2max = 5;
% Pareto weight
alpha = 0.5;
%}

% utility function
u1 = @(x,Q) a1.*log(x) + (1-a1).*log(Q);
u2 = @(x,Q) a2.*log(x) + (1-a2).*log(Q);
%u1 = @(x,Q) a1.*x + (1-a1).*Q;
%u2 = @(x,Q) a2.*x + (1-a2).*Q;


%% Grid set-up for Y (income), S (Sharing rule) and W (continuation value)

% y1: income of agent 1
prob_y1 = ones(1,n).*n.^(-1);
y1_grid = linspace(y1min,y1max,n);
% y2: income of agent 2
prob_y2 = ones(1,n).*n.^(-1);
y2_grid = linspace(y2min,y2max,n);
% s: sharing rule
% first divide total endowment by sharing rule, then each agent maximizes
% own utility
s_grid = linspace(0.1,0.9,ns);
% w: continuation value of agent 2
% turn on or off participation constraint
pc = 0;
w_min = (a2*log(a2*y2min) + (1-a2)*log((1-a2)*y2min))/(1-beta);
w_aut = mean(a2*log(a2*y2_grid) + (1-a2)*log((1-a2)*y2_grid))/(1-beta);
w_max = (a2*log(a2*y2max) + (1-a2)*log((1-a2)*y2max))/(1-beta);
if pc == 1
% with participation constraint, continuation value cannot be lower than
% autarky value
    w_low = linspace(w_aut,(w_aut+w_max)/2,nw/4);
    w_high = linspace((w_aut+w_max)/2,w_max,3*nw/4);
    w_grid = [w_low w_high];
else
    w_low = linspace(w_min,(w_min+w_max)/2,nw/4);
    w_high = linspace((w_min+w_max)/2,w_max,3*nw/4);
    w_grid = [w_low w_high];
end

% P: continuation value of agent 1
% with participation constraint, continuation value cannot be lower than
% autarky value
P_aut = mean(a1*log(a1*y1_grid) + (1-a1)*log((1-a1)*y1_grid))/(1-beta);

%% Iterations

% Space of feasible w gridpoints, starting with the entire w-grid
exitflag = ones(1, nw);
w_feasible = w_grid(exitflag==1);
nfeas = length(w_feasible);

% initial guess of P function
P = fliplr(w_feasible);
P_update = P;
pp = kron(ones(1, n*n*ns), P);

% set up for convergence
diff_feas = 100;
error = 0.01;

while diff_feas>error
    
    tic
      
    % vector to store optimization flags
    exitflag = ones(1,nfeas);
    
    % A matrix to store probabilities
    X_all = zeros(nfeas*ns*n*n,nfeas);
    lambda_lower = zeros(nfeas*ns*n*n,nfeas);
    lambda_upper = zeros(nfeas*ns*n*n,nfeas);
    lambda_eqlin = zeros(n*n+2,nfeas);
    lambda_ineqlin = zeros(2*n*(n-1),nfeas);
            
    %% Constraints (that do not depend on the P function)
            % Upper bounds and lower bounds (between 0 and 1)
            UB=ones(n*n*ns*nfeas,1); %the vector of upper bounds
            LB=zeros(n*n*ns*nfeas,1); %lower bounds
            % setting up kronecker product
            yy1 = kron(y1_grid, ones(1, ns*nfeas*n));
            yy2 = kron(ones(1,n), kron(y2_grid, ones(1, ns*nfeas)));
            ss = kron(ones(1,n*n), kron(s_grid, ones(1, nfeas)));
            ww = kron(ones(1, n*n*ns), w_feasible);
            
            % calculate consumption
            cc1 = a1.*ss.*(yy1+yy2);
            cc2 = a2.*(1-ss).*(yy1+yy2);
            QQ = (yy1+yy2-cc1-cc2)./price;
            
            % agent 1's consumption should not be negative
            neg = (QQ<=0);
            UB(neg) = 0;
            
            % sum of probabilities
            Aeq1=ones(1, n*n*ns*nfeas); %the coefficients are ones on each π
            beq1=1; %the sum of probabilities needs to be 1.
            
            % incentive compatibility for agent 2
            counter = 0;
            Aineq1 = zeros(n*(n-1),n*n*ns*nfeas);
            for i = 1:n % each state of the world of agent 2
                yi_index = (yy2 == y2_grid(i));
                alt = [1:i-1 i+1:n]; % all other states of the world
                for j = 1:n-1 % each alternate state of the world
                    counter = counter+1;
                    yj_index = (yy2 == y2_grid(alt(j)));
                    Aineq1(counter,yi_index) = -(u2(cc2(yi_index),QQ(yi_index))+ beta.*ww(yi_index))./prob_y2(i);
                    Aineq1(counter,yj_index) = (u2(cc2(yj_index),QQ(yj_index))+ beta.*ww(yj_index))./prob_y2(alt(j));
                    ic_neg = false(1,n*n*ns*nfeas);
                    ic_neg(yj_index) = (QQ(yi_index)<=0);
                    Aineq1(counter,neg) = 0;
                    Aineq1(counter,ic_neg) = 0;
                end
            end
            bineq1 = zeros(1,n*(n-1));
            
            % conditional probabilities add up to unconditional probabilities of y
            Aeq3 = kron(eye(n*n),ones(1,nfeas*ns));
            beq3 = kron(prob_y1,prob_y2)';
            
    %% P function iteration
    diff = 100;
    
    while diff>error
        % for each v
        for k = 1:nfeas
            
            v = w_feasible(k);
            
            %% Constraints that depend on v and P(v)
            % incentive compatibility for agent 1
            counter = 0;
            Aineq2 = zeros(n*(n-1),n*n*ns*nfeas);
            for i = 1:n % each state of the world of agent 1
                yi_index = (yy1 == y1_grid(i));
                alt = [1:i-1 i+1:n]; % all other states of the world
                for j = 1:n-1 % each alternate state of the world
                    counter = counter+1;
                    yj_index = (yy1 == y1_grid(alt(j)));
                    Aineq2(counter,yi_index) = -(u1(cc1(yi_index),QQ(yi_index)) + beta.*pp(yi_index))./prob_y1(i);
                    Aineq2(counter,yj_index) = (u1(cc1(yj_index),QQ(yj_index)) + beta.*pp(yj_index))./prob_y1(alt(j));
                    ic_neg = false(1,n*n*ns*nfeas);
                    ic_neg(yj_index) = (QQ(yi_index)<=0);
                    Aineq2(counter,neg) = 0;
                    Aineq2(counter,ic_neg) = 0;
                end
            end
            bineq2 = zeros(1,n*(n-1));
            
            % promise keeping constraint (2)
            Aeq2 = u2(cc2, QQ) + beta.*ww;
            Aeq2(neg) = 0;
            beq2 = v;
            
            %% All constraints
            Aeq=[Aeq1; Aeq2; Aeq3]; %matrix of coefficients on the equality constraints
            beq=[beq1; beq2; beq3]; %intercepts
            Aineq = [Aineq1; Aineq2];
            bineq = [bineq1; bineq2];
            
            %% Objective function
            f = - u1(cc1, QQ) - beta.*pp;
            f(neg) = 0;
            
            %% Optimization using linprog
            options = optimset('Display','off','TolFun', 10^-3,'MaxIter',10000,'TolX', 10^-3);
            [X, fval, exitflag(k), output, lambda] = linprog(f, Aineq, bineq, Aeq, beq, LB, UB,[],options);
            
            % If optimization was conducted successfully
            if exitflag(k) == 1
                % save X matrix
                X_all(:,k) = X;
                
                % save Lagrange multipliers
                lambda_lower(:,k) = lambda.lower;
                lambda_upper(:,k) = lambda.upper;
                lambda_eqlin(:,k) = lambda.eqlin;
                lambda_ineqlin(:,k) = lambda.ineqlin;
                
                % update P
                P_update(k) = -fval;
            end
        end
        
        % Check convergence of P function
        diff = max(abs(P-P_update))
        
        % Update P function
        P = P_update;
        pp = kron(ones(1, n*n*ns), P);
    end
    
    toc 
    
    %% Check feasibility
    % agent 1's participation constraint
    P_feasible = (P>=P_aut);
    % optimization was feasible
    exitflag_feasible = (exitflag==1);
    % both constraints
    if pc == 1
        w_update = exitflag_feasible & P_feasible;
    else
        w_update = exitflag_feasible;
    end
    
    % Check convergence of w-space
    diff_feas = max(abs(w_update-1));
    
    % Update w-space
    w_feasible = w_feasible(w_update);
    nfeas = length(w_feasible);
    
    % Update initial guess of P function
    P = P(w_update);
    P_update = P;
    pp = kron(ones(1, n*n*ns), P);
    
end

%% Use Pareto weight to find initial values
% Maximize social planner's expected utility at period 0
social_planner = alpha.*P + (1-alpha).*w_feasible;
[w0, w0_index] = max(social_planner);

%

%% plot P function
figure;
plot(w_feasible,P);
title('Agent 1 Utility as a Function of Agent 2 Utility');

%% Display probabilities
for i = 1:length(w_feasible)
    xp=find(X_all(:,i)>10^-4); %gives the indices of all elements of X > 10ˆ-4
    disp(['v =',num2str(w_feasible(i))]);
    disp('y1 y2 c1 c2 Q w prob')
    disp('———————————————————')
    disp([yy1(xp)', yy2(xp)', cc1(xp)', cc2(xp)', QQ(xp)', ww(xp)', X_all(xp,i)]);
end


%% Calculate transition probabilities
% for each promised utility value (this period)
w_trans = zeros(nfeas);
% calculate probability of consumption
conscdf1 = zeros(n*n*nfeas*ns,nfeas);
conscdf2 = zeros(n*n*nfeas*ns,nfeas);
Qcdf = zeros(n*n*nfeas*ns,nfeas);
[cons1, cons_index1] = sort(cc1);
[cons2, cons_index2] = sort(cc2);
[qcons, q_index] = sort(QQ);

for i = 1:nfeas
    pi = X_all(:,i);
    % for each next period value
    for j = 1:nfeas
        index = (ww == w_feasible(j));
        w_trans(i,j) = sum(pi(index));
    end
    % consumption cdf: first sort the probabiilities according to magnitude
    % (pdf), then add up to get cdf
    % private consumption of agent 1
    pi_cons1 = pi(cons_index1);
    conscdf1(:,i) = cumsum(pi_cons1);
    % private consumption of agent 2
    pi_cons2 = pi(cons_index2);
    conscdf2(:,i) = cumsum(pi_cons2);
    % public good consumption
    pi_q = pi(q_index);
    Qcdf(:,i) = cumsum(pi_q);
end

%% Stationary distribution of Markov chain
% find the eigenvectors and eigenvalues
[V,D] = eig(w_trans');
% find where eigenvalue = 1
eig1_index = find(imag(D)==0 & (abs(real(D)-1)<0.0001));
if isempty(eig1_index) == true
    disp('There is no stationary distribution')
else
    [row,col]=ind2sub([nfeas nfeas],eig1_index);
    % calculate probability distribution
    invariant_w = V(:,row(1))./sum(V(:,row(1)));
    % probability of P(w)
    prob_p = sort([P' invariant_w]);
    prob_p(:,2) = cumsum(prob_p(:,2));
    % plot cdf
    figure;
    subplot(1,2,1);
    plot(w_feasible, cumsum(invariant_w));
    title('Invariant Distribution of agent 2 utility')
    subplot(1,2,2);
    plot(prob_p(:,1), prob_p(:,2));
    title('Invariant Distribution of agent 1 utility')
end

%% Plot distribution of consumption
figure;
subplot(1,3,1);
plot(cons1,conscdf1(:,1),cons1,conscdf1(:,10),cons1,conscdf1(:,20),cons1,conscdf1(:,26))
legend(['v=',num2str(w_feasible(1))],['v=',num2str(w_feasible(10))],['v=',num2str(w_feasible(20))],['v=',num2str(w_feasible(26))],'Location','Southeast');
xlim([0 y1max]);
ylim([0 1]);
title('Agent 1 Private Consumption');

subplot(1,3,2);
plot(cons2,conscdf2(:,1),cons2,conscdf2(:,10),cons2,conscdf2(:,20),cons2,conscdf2(:,26))
legend(['v=',num2str(w_feasible(1))],['v=',num2str(w_feasible(10))],['v=',num2str(w_feasible(20))],['v=',num2str(w_feasible(26))],'Location','Southeast');
xlim([0 y2max]);
ylim([0 1]);
title('Agent 2 Private Consumption');

subplot(1,3,3);
plot(qcons,Qcdf(:,1),qcons,Qcdf(:,10),qcons,Qcdf(:,20),qcons,Qcdf(:,20))
legend(['v=',num2str(w_feasible(1))],['v=',num2str(w_feasible(10))],['v=',num2str(w_feasible(20))],['v=',num2str(w_feasible(26))],'Location','Southeast');
xlim([0 y2max]);
ylim([0 1]);
title('Public Good Consumption');

%% Simulation
% simulation time periods
T = 1000;

% generate alternate income path
ind_path1 = repmat([1 2],1,T/2);
ind_path2 = repmat([2 1],1,T/2);

% Simulate income processes
%ind_path1 = randi([1 n],T,1);
%ind_path2 = randi([1 n],T,1);
y1_path = y1_grid(ind_path1);
y2_path = y2_grid(ind_path2);

% matrices
c1_ic_path = zeros(1,T);
c2_ic_path = zeros(1,T);
Q_ic_path = zeros(1,T);
w_ic_path = zeros(1,T+1);
w_index = zeros(1,T+1);
s_path = zeros(1,T);

% kronecker products
y1_grid = linspace(y1min,y1max,n);
yy1 = kron(y1_grid, ones(1, ns*nfeas*n));
y2_grid = linspace(y2min,y2max,n);
yy2 = kron(ones(1,n), kron(y2_grid, ones(1, ns*nfeas)));
s_grid = linspace(0.1,0.9,ns);
ss = kron(ones(1,n*n), kron(s_grid, ones(1, nfeas)));
ww = kron(ones(1, n*n*ns), w_feasible);
cc1 = a1.*ss.*(yy1+yy2);
cc2 = a2.*(1-ss).*(yy1+yy2);
QQ = (yy1+yy2-cc1-cc2)./price;

% beginning w
w_ic_path(1) = w0;
[~, w_index(1)] = min(abs(w0-w_feasible));

% in each period
for t = 1:T
    % income realization
    y1ind = (yy1 == y1_path(t));
    y2ind = (yy2 == y2_path(t));
    yind = y1ind & y2ind;
    w_realized = ww(yind);
    y1_realized = yy1(yind);
    y2_realized = yy2(yind);
    c1_realized = cc1(yind);
    c2_realized = cc2(yind);
    Q_realized = QQ(yind);
    s_realized = ss(yind);
    % probability vector is determined by last period's future promised
    % utility and this period's income realization
    pi = X_all(yind,w_index(t));
    picum = cumsum(pi).*(n.^2);
    % realization of income, transfer and continuation value
    randnum = rand(1); % random number
    crit1 = (picum > randnum); % random number is less than cumulative distribution
    crit2 = (pi~=0); % points that have mass
    % values that fit criterion
    w_next = w_realized(crit1 & crit2);
    y1_all = y1_realized(crit1 & crit2);
    y2_all = y2_realized(crit1 & crit2);
    c1_all = c1_realized(crit1 & crit2);
    c2_all = c2_realized(crit1 & crit2);
    Q_all = Q_realized(crit1 & crit2);
    s_all = s_realized(crit1 & crit2);
    % update values
    w_ic_path(t+1) = w_next(1);
    c1_ic_path(t) = c1_all(1);
    c2_ic_path(t) = c2_all(1);
    Q_ic_path(t) = Q_all(1);
    s_path(t) = s_all(1);
    % index
    [~, w_index(t+1)] = min(abs(w_feasible-w_ic_path(t+1)));
end

P_ic_path = P(w_index);

% utility path
u1_ic_path = u1(c1_ic_path,Q_ic_path);
u2_ic_path = u2(c2_ic_path,Q_ic_path);

%% Create Graph
figure;
subplot(4,2,1);
plot(1:T,y1_path,1:T,y2_path);
title('Income')

subplot(4,2,2);
plot(1:T,c1_ic_path);
title('C1')

subplot(4,2,3);
plot(1:T,c2_ic_path);
title('C2')

subplot(4,2,4);
plot(1:T,Q_ic_path);
title('Q')

subplot(4,2,5);
plot(1:T,w_ic_path(1:T));
title('w')

subplot(4,2,6);
plot(1:T,P_ic_path(1:T));
title('P')

subplot(4,2,7);
plot(1:T,u1_ic_path);
title('Utility - Agent 1')

subplot(4,2,8);
plot(1:T,u2_ic_path);
title('Utility - Agent 2')

%}
    