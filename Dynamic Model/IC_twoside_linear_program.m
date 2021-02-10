%% Two side IC
% this code solves the infinite horizon two-side IC problem with two agents
% and private consumption only. The outer loop converges on the possible
% utility gridpoints for agent 2 (see Wang 1995), the inner loop converges
% on P function that maps utility of agent 2 to utility of agent 1.
% Within the inner loop, maximization is done using linprog with a
% probability vector (see SL exercise 20.4 and Karaivanov's Matlab code for
% moral hazard).
% Feb 4: The w-transition matrix is still a little weird (not clean like
% one-side IC), but with enough grid points and simulation periods,
% expected consumption of agent 1 and agent 2 seem to converge

clear all; close all; clc;

%% Paramters set-up
% discounting factor
beta = 0.94;
% risk aversion
gamma = 3;
% utility function
u = @(x) x.^(1-gamma)./(1-gamma);

%% Whether to turn on participation constraint
% set pc = 0 to turn off participation constraint
pc = 1;

%% Grid set-up for Y (income), B (transfer) and W (continuation value)
% number of gridpoints
s1 = 3;
s2 = 3;
nb = 51;
nw = 51;
% y1: income of agent 1
prob_y1 = ones(1,s1).*s1.^(-1);
y1_min = 1;
y1_max = 10;
y1_grid = linspace(y1_min,y1_max,s1);
% y2: income of agent 2
prob_y2 = ones(1,s2).*s2.^(-1);
y2_min = 1;
y2_max = 10;
y2_grid = linspace(y2_min,y2_max,s2);
% b: transfer from agent 1 to agent 2
% b cannot be larger than agent 1's endowment and cannot be smaller than
% negative agent 2's endowment
b_min = -y2_max;
b_max = y1_max;
b_grid = linspace(b_min,b_max,nb);

% w: continuation value of agent 2
w_min = y2_min.^(1-gamma)./((1-beta).*(1-gamma));
w_aut = mean(y2_grid.^(1-gamma)./((1-beta).*(1-gamma)));
w_max = y2_max.^(1-gamma)./((1-beta).*(1-gamma));

% no participation constraint
if pc == 0
    w_grid = linspace(w_min,w_max,nw);
end

% with participation constraint, continuation value cannot be lower than
% autarky value
if pc == 1
    w_grid = linspace(w_aut,w_max,nw);
end

% P: continuation value of agent 1
% with participation constraint, continuation value cannot be lower than
% autarky value
P_aut = mean(y1_grid.^(1-gamma)./((1-beta).*(1-gamma)));

%% Iteration on feasible w-space, then on P-function

% Space of feasible w gridpoints, starting with the entire w-grid
exitflag = ones(1, nw);
w_feasible = w_grid(exitflag==1);
nfeas = length(w_feasible);

% initial guess of P function
P = fliplr(w_feasible);
P_update = P;
pp = kron(ones(1, s1*s2*nb), P);

% set up for convergence
diff_feas = 100;
error = 0.01;

%% outer loop: iterate on the feasible space of w gridpoints
while diff_feas>error
    
    % vector to store optimization flags
    exitflag = ones(1,nfeas);
    
    % A matrix to store probabilities
    X_all = zeros(nfeas*nb*s1*s2,nfeas);
    
    %% Constraints that don't depend on v or P(v)
    % Upper bounds and lower bounds (between 0 and 1)
    UB=ones(s1*s2*nb*nfeas,1); %the vector of upper bounds
    LB=zeros(s1*s2*nb*nfeas,1); %lower bounds
    % setting up kronecker product
    yy1 = kron(y1_grid, ones(1, nb*nfeas*s2));
    yy2 = kron(ones(1,s1), kron(y2_grid, ones(1, nb*nfeas)));
    bb = kron(ones(1, s1*s2), kron(b_grid, ones(1, nfeas)));
    ww = kron(ones(1, s1*s2*nb), w_feasible);
    % agent 1's consumption should not be negative
    neg1 = (yy1-bb<=0);
    UB(neg1) = 0;
    % agent 2's consumption should not be negative
    neg2 = (yy2+bb<=0);
    UB(neg2) = 0;
    
    % sum of probabilities
    Aeq1=ones(1, s1*s2*nb*nfeas); %the coefficients are ones on each π
    beq1=1; %the sum of probabilities needs to be 1.
    
    % incentive compatibility for agent 2
    counter = 0;
    Aineq1 = zeros(s2*(s2-1),s1*s2*nb*nfeas);
    for i = 1:s2 % each state of the world of agent 2
        yi = y2_grid(i); % income
        yi_index = (yy2 == y2_grid(i));
        alt = [1:i-1 i+1:s2]; % all other states of the world
        for j = 1:s2-1 % each alternate state of the world
            counter = counter+1;
            yj_index = (yy2 == y2_grid(alt(j)));
            Aineq1(counter,yi_index) = -(u(yi+bb(yi_index))+ beta.*ww(yi_index))./prob_y2(i);
            Aineq1(counter,yj_index) = (u(yi+bb(yi_index))+ beta.*ww(yi_index))./prob_y2(alt(j));
            ic_neg = false(1,s1*s2*nb*nfeas);
            ic_neg(yj_index) = (yi+bb(yi_index)<=0);
            Aineq1(counter,neg1) = 0;
            Aineq1(counter,neg2) = 0;
            Aineq1(counter,ic_neg) = 0;
        end
    end
    bineq1 = zeros(1,s2*(s2-1));
    
    % conditional probabilities add up to unconditional probabilities of y
    Aeq3 = kron(eye(s1*s2),ones(1,nfeas*nb));
    beq3 = kron(prob_y1,prob_y2)';
    
    %% inner loop: iterate on P function until convergence
    diff = 100;

    while diff>error
        
        % for each v
        for k = 1:nfeas
            
            v = w_feasible(k);
            
            %% Constraints that depend on v and P(v)
            % incentive compatibility for agent 1
            counter = 0;
            Aineq2 = zeros(s1*(s1-1),s1*s2*nb*nfeas);
            for i = 1:s1 % each state of the world of agent 1
                yi = y1_grid(i); % income
                yi_index = (yy1 == y1_grid(i));
                alt = [1:i-1 i+1:s1]; % all other states of the world
                for j = 1:s1-1 % each alternate state of the world
                    counter = counter+1;
                    yj_index = (yy1 == y1_grid(alt(j)));
                    Aineq2(counter,yi_index) = -(u(yi-bb(yi_index)) + beta.*pp(yi_index))./prob_y1(i);
                    Aineq2(counter,yj_index) = (u(yi-bb(yi_index)) + beta.*pp(yi_index))./prob_y1(alt(j));
                    ic_neg = false(1,s1*s2*nb*nfeas);
                    ic_neg(yj_index) = (yi-bb(yi_index)<=0);
                    Aineq2(counter,neg1) = 0;
                    Aineq2(counter,neg2) = 0;
                    Aineq2(counter,ic_neg) = 0;
                end
            end
            bineq2 = zeros(1,s1*(s1-1));
            
            % promise keeping constraint (2)
            Aeq2 = (yy2 + bb).^(1-gamma)./(1-gamma) + beta.*ww;
            Aeq2(neg1) = 0;
            Aeq2(neg2) = 0;
            beq2 = v;
            
            %% all constraints
            Aeq=[Aeq1; Aeq2; Aeq3]; %matrix of coefficients on the equality constraints
            beq=[beq1; beq2; beq3]; %intercepts
            Aineq = [Aineq1; Aineq2];
            bineq = [bineq1; bineq2];
            
            %% Objective function
            f = - (yy1 - bb).^(1-gamma)./(1-gamma) - beta.*pp;
            f(neg1) = 0;
            f(neg2) = 0;
            
            %% using linprog to solve for maximization
            options = optimset('Display','off','TolFun', 10^-3,'MaxIter',10000,'TolX', 10^-3);
            [X, fval, exitflag(k)] = linprog(f, Aineq, bineq, Aeq, beq, LB, UB,[],options);
            
            % if linprog exited successfully
            if exitflag(k) == 1
                % save X matrix
                X_all(:,k) = X;
                
                % update P
                P_update(k) = -fval;
            end
        end
        
        % Check convergence of P function
        diff = max(abs(P-P_update))
        
        % Update P function
        P = P_update;
        pp = kron(ones(1, s1*s2*nb), P);
    end
    
    %% Check feasibility
    % agent 1's participation constraint
    P_feasible = (P>=P_aut);
    % optimization was feasible
    exitflag_feasible = (exitflag==1);
    
    % no participation constraint
    if pc == 0
        w_update = exitflag_feasible;
    end
    
    % participation constraint
    if pc == 1
        w_update = P_feasible & exitflag_feasible;
    end
    
    % Check convergence of w-space
    diff_feas = max(abs(w_update-1));
    
    % Update w-space
    w_feasible = w_feasible(w_update);
    nfeas = length(w_feasible);
    
    % Update initial guess of P function
    P = P(w_update);
    P_update = P;
    pp = kron(ones(1, s1*s2*nb), P);

end

%% Display program solutions
% for each value of v
for i = 1:length(w_feasible)
    iw = find(w_grid == w_feasible(i));
    % find probability mass points
    xp=find(X_all(:,iw)>10^-4);
    % display on screen
    disp(['v =',num2str(w_grid(i))]);
    disp('y1 y2 b w prob')
    disp('———————————————————')
    disp([yy1(xp)', yy2(xp)', bb(xp)', ww(xp)', X_all(xp,iw)]);
end

%% Calculate transition probabilities
% for each promised utility value (this period)
w_trans = zeros(nfeas);
for i = 1:nfeas
    pi = X_all(:,i);
    % for each next period value
    for j = 1:nfeas
        index = (ww == w_grid(j));
        w_trans(i,j) = sum(pi(index));
    end
end

%% Stationary distribution of Markov chain
% find the eigenvectors, the first one is with eigenvalue 1
[V,D] = eig(w_trans');
% find where eigenvalue = 1
eig1_index = find(imag(D)==0 & (abs(real(D)-1)<0.001));

if eig1_index == []
    disp('There is no stationary distribution')
else
    [row,col]=ind2sub([nfeas nfeas],eig1_index);
    % calculate probability distribution
    invariant_w = V(:,row)./sum(V(:,row));
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

% plot P function
figure;
plot(w_feasible,P);
title('Agent 1 Utility as a Function of Agent 2 Utility');

%% Use Pareto weight to find initial values
% Pareto weight
alpha = 0.5;
% Maximize social planner's expected utility at period 0
social_planner = alpha.*P + (1-alpha).*w_feasible;
[w0, w0_index] = max(social_planner);

%% Simulation
% set seed
rng(6);
% number of periods
T = 5000;
% matrices
y1_sim = zeros(1,T);
y2_sim = zeros(1,T);
b_sim = zeros(1,T);
w_sim = zeros(1,T+1);
w_index = zeros(1,T+1);
conscdf1 = zeros(s1*s2*nfeas*nb,T+1);
conscdf2 = zeros(s1*s2*nfeas*nb,T+1);
% beginning w
w_sim(1) = w0;
w_index(1) = w0_index;

% in each period
for t = 1:T
    % probability vector is determined by last period's future promised
    % utility and this period's income realization
    pi = X_all(:,w_index(t));
    picum = cumsum(pi);
    % calculate probability of consumption
    [cons1, cons_index1] = sort(yy1-bb);
    pi_cons1 = pi(cons_index1);
    conscdf1(:,t) = cumsum(pi_cons1);
    % calculate probability of consumption
    [cons2, cons_index2] = sort(yy2+bb);
    pi_cons2 = pi(cons_index2);
    conscdf2(:,t) = cumsum(pi_cons2);
    % realization of income, transfer and continuation value
    randnum = rand(1); % random number
    crit1 = (picum > randnum); % random number is less than cumulative distribution
    crit2 = (pi~=0); % points that have mass
    % values that fit criterion
    w_next = ww(crit1 & crit2);
    y1_all = yy1(crit1 & crit2);
    y2_all = yy2(crit1 & crit2);
    b_all = bb(crit1 & crit2);
    % update values
    w_sim(t+1) = w_next(1);
    y1_sim(t) = y1_all(1);
    y2_sim(t) = y2_all(1);
    b_sim(t) = b_all(1);
    % index
    [~, w_index(t+1)] = min(abs(w_feasible-w_sim(t+1)));
end

%% Plot simulation
figure;
% income - agent 1
subplot(2,3,1);
plot(1:T,y1_sim(1:T));
ylim([0 12]);
title('Income - Agent 1');
% income - agent 2
subplot(2,3,4);
plot(1:T,y2_sim(1:T));
ylim([0 12]);
title('Income - Agent 2');
% consumption - agent 1
subplot(2,3,2);
c1_sim = y1_sim-b_sim;
plot(1:T,c1_sim(1:T));
ylim([0 12]);
title('Consumption - Agent 1');
% consumption - agent 2
subplot(2,3,5);
c2_sim = y2_sim+b_sim;
plot(1:T,c2_sim(1:T));
ylim([0 12]);
title('Consumption - Agent 2');
% transfer
subplot(2,3,3);
plot(1:T,b_sim(1:T));
ylim([-6 6]);
title('Transfer');
% continuation value of agent 2
subplot(2,3,6);
P_sim = P(w_index);
plot(1:T,P_sim(1:T),1:T,w_sim(1:T));
legend('Agent 1','Agent 2','Location','Southeast')
title('Continuation Value');

%% Empirical cdf plot
figure;
% income 
subplot(2,2,1);
cdfplot(y1_sim);
hold on
cdfplot(y2_sim);
title('Income');
legend('Agent 1','Agent 2','Location','Southeast')
% consumption
subplot(2,2,2);
cdfplot(c1_sim);
hold on
cdfplot(c2_sim);
title('Consumption');
% transfer
subplot(2,2,3);
cdfplot(b_sim);
title('Transfer');
% utility
subplot(2,2,4);
cdfplot(P_sim);
hold on
cdfplot(w_sim);
title('Continuation Value');


%% Correlations

% income of two agents are not correlated
corrcoef(y1_sim(1:T), y2_sim(1:T));

% income and consumption of the same agent is correlated
corrcoef(y1_sim(1:T), c1_sim(1:T));
corrcoef(y2_sim(1:T), c2_sim(1:T));

% consumption of two agents are correlated
corrcoef(c1_sim(1:T), c2_sim(1:T));

% mean transfer
mean(b_sim(1:T))

% on average, agent 1 consumes more
mean(c1_sim(1:T) - c2_sim(1:T))

%% Plot distribution of consumption
figure;
subplot(1,2,1);
plot(cons1,conscdf1(:,5),cons1,conscdf1(:,10),cons1,conscdf1(:,25),cons1,conscdf1(:,100))
legend('t=5','t=10','t=25','t=100');
xlim([0 y1_max]);
ylim([0 1]);
title('Agent 1');

subplot(1,2,2);
plot(cons2,conscdf2(:,5),cons2,conscdf2(:,10),cons2,conscdf2(:,25),cons2,conscdf2(:,100))
legend('t=5','t=10','t=25','t=100');
xlim([0 y2_max]);
ylim([0 1]);
title('Agent 2');
