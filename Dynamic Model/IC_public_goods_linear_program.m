%% Two side IC with public goods (public good as state variable)
% February 2021
% This was my first attempt at including public goods. I included both
% transfer from agent 1 to agent 2 and public good consumption as state
% variables, in addition to agent 2's promised utility. Three state
% variables means the code is significantly slower, and later I switched to
% only one state variable for current period allocation.
% this code solves the infinite horizon two-side IC problem with two agents
% and private consumption and public good consumption. The outer loop converges on the possible
% utility gridpoints for agent 2 (see Wang 1995), the inner loop converges
% on P function that maps utility of agent 2 to utility of agent 1.
% Within the inner loop, maximization is done using linprog with a
% probability vector (see SL exercise 20.4 and Karaivanov's Matlab code for
% moral hazard).

% set up
clear all; close all; clc;
beta = 0.94;
price = 1;
a1 = 0.5;
a2 = 0.5;

% utility function
u1 = @(x,Q) a1.*log(x) + (1-a1).*log(Q);
u2 = @(x,Q) a2.*log(x) + (1-a2).*log(Q);

% grid for Y (income), B (transfer), W (continuation value) and Q (public
% goods)
s1 = 2;
s2 = 2;
nb = 15;
nw = 15;
nq = 5;
prob_y1 = ones(1,s1).*s1.^(-1);
prob_y2 = ones(1,s2).*s2.^(-1);
y1_min = 1;
y1_max = 10;
y2_min = 1;
y2_max = 10;
%w_min = (a2*log(a2*y2_min) + (1-a2)*log((1-a2)*y2_min))/(1-beta);
w_min = 0.5.*(a2*log(a2*y2_min) + (1-a2)*log((1-a2)*y2_min))/(1-beta) + 0.5.*(a2*log(a2*y2_max) + (1-a2)*log((1-a2)*y2_max))/(1-beta);
w_max = (a2*log(a2*y2_max) + (1-a2)*log((1-a2)*y2_max))/(1-beta);
b_min = -y2_max;
b_max = y1_max;
Q_max = (y1_max + y2_max)./price;
y1_grid = linspace(y1_min,y1_max,s1);
y2_grid = linspace(y2_min,y2_max,s2);
b_grid = linspace(b_min,b_max,nb);
w_grid = linspace(w_min,w_max,nw);
Q_grid = linspace(1e-2,Q_max,nq);


%% Iterations

% Space of feasible w gridpoints, starting with the entire w-grid
exitflag = ones(1, nw);
w_feasible = w_grid(exitflag==1);
nfeas = length(w_feasible);

% initial guess of P function
P = fliplr(w_feasible);

% set up for convergence
diff_feas = 100;
error = 0.01;

for l = 1:2
    
    tic
    
    % Initial guess of P function. From second iteration, use the
    % previously found P function and constrain it to the new set of
    % w-gridpoints.
    P = P(exitflag==1);
    P_update = P;
    pp = kron(ones(1, s1*s2*nb*nq), P);
    
    % vector to store optimization flags
    exitflag = ones(1,nfeas);
    
    % A matrix to store probabilities
    X_all = zeros(nfeas*nb*nq*s1*s2,nfeas);
            
    %% Constraints (that do not depend on the P function)
            % Upper bounds and lower bounds (between 0 and 1)
            UB=ones(s1*s2*nb*nfeas*nq,1); %the vector of upper bounds
            LB=zeros(s1*s2*nb*nfeas*nq,1); %lower bounds
            % setting up kronecker product
            yy1 = kron(y1_grid, ones(1, nb*nfeas*s2*nq));
            yy2 = kron(ones(1,s1), kron(y2_grid, ones(1, nb*nfeas*nq)));
            QQ = kron(ones(1,s1*s2), kron(Q_grid, ones(1, nb*nfeas)));
            bb = kron(ones(1,s1*s2*nq), kron(b_grid, ones(1, nfeas)));
            ww = kron(ones(1, s1*s2*nb*nq), w_feasible);
            
            % agent 1's consumption should not be negative
            neg1 = (yy1-bb-price.*QQ<=0);
            UB(neg1) = 0;
            % agent 2's consumption should not be negative
            neg2 = (yy2+bb-price.*QQ<=0);
            UB(neg2) = 0;
            
            % sum of probabilities
            Aeq1=ones(1, s1*s2*nb*nfeas*nq); %the coefficients are ones on each π
            beq1=1; %the sum of probabilities needs to be 1.
            
            % incentive compatibility for agent 2
            counter = 0;
            Aineq1 = zeros(s2*(s2-1),s1*s2*nb*nfeas*nq);
            for i = 1:s2 % each state of the world of agent 2
                yi = y2_grid(i); % income
                yi_index = (yy2 == y2_grid(i));
                alt = [1:i-1 i+1:s2]; % all other states of the world
                for j = 1:s2-1 % each alternate state of the world
                    counter = counter+1;
                    yj_index = (yy2 == y2_grid(alt(j)));
                    Aineq1(counter,yi_index) = -(u2(yi+bb(yi_index)-price.*QQ(yi_index),QQ(yi_index))+ beta.*ww(yi_index))./prob_y2(i);
                    Aineq1(counter,yj_index) = (u2(yi+bb(yi_index)-price.*QQ(yi_index),QQ(yi_index))+ beta.*ww(yi_index))./prob_y2(alt(j));
                    ic_neg = false(1,s1*s2*nb*nfeas*nq);
                    ic_neg(yj_index) = (yi+bb(yi_index)-price.*QQ(yi_index)<=0);
                    Aineq1(counter,neg1) = 0;
                    Aineq1(counter,neg2) = 0;
                    Aineq1(counter,ic_neg) = 0;
                end
            end
            bineq1 = zeros(1,s2*(s2-1));
            
            % conditional probabilities add up to unconditional probabilities of y
            Aeq3 = kron(eye(s1*s2),ones(1,nfeas*nb*nq));
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
            Aineq2 = zeros(s1*(s1-1),s1*s2*nb*nfeas*nq);
            for i = 1:s1 % each state of the world of agent 1
                yi = y1_grid(i); % income
                yi_index = (yy1 == y1_grid(i));
                alt = [1:i-1 i+1:s1]; % all other states of the world
                for j = 1:s1-1 % each alternate state of the world
                    counter = counter+1;
                    yj_index = (yy1 == y1_grid(alt(j)));
                    Aineq2(counter,yi_index) = -(u1(yi-bb(yi_index)-price.*QQ(yi_index),QQ(yi_index)) + beta.*pp(yi_index))./prob_y1(i);
                    Aineq2(counter,yj_index) = (u1(yi-bb(yi_index)-price.*QQ(yi_index),QQ(yi_index)) + beta.*pp(yi_index))./prob_y1(alt(j));
                    ic_neg = false(1,s1*s2*nb*nfeas*nq);
                    ic_neg(yj_index) = (yi-bb(yi_index)-price.*QQ(yi_index)<=0);
                    Aineq2(counter,neg1) = 0;
                    Aineq2(counter,neg2) = 0;
                    Aineq2(counter,ic_neg) = 0;
                end
            end
            bineq2 = zeros(1,s1*(s1-1));
            
            % promise keeping constraint (2)
            Aeq2 = u2(yy2 + bb - price.*QQ, QQ) + beta.*ww;
            Aeq2(neg1) = 0;
            Aeq2(neg2) = 0;
            beq2 = v;
            
            %% All constraints
            Aeq=[Aeq1; Aeq2; Aeq3]; %matrix of coefficients on the equality constraints
            beq=[beq1; beq2; beq3]; %intercepts
            Aineq = [Aineq1; Aineq2];
            bineq = [bineq1; bineq2];
            
            %% Objective function
            f = - u1(yy1 - bb - price.*QQ, QQ) - beta.*pp;
            f(neg1) = 0;
            f(neg2) = 0;
            
            %% Optimization using linprog
            options = optimset('Display','off','TolFun', 10^-3,'MaxIter',10000,'TolX', 10^-3);
            [X, fval, exitflag(k)] = linprog(f, Aineq, bineq, Aeq, beq, LB, UB,[],options);
            
            % If optimization was conducted successfully
            if exitflag(k) == 1
                % save X matrix
                X_all(:,k) = X;
                
                % update P
                P_update(k) = -fval;
            end
        end
        
        % Check convergence of P function
        diff = max(abs(P-P_update));
        
        % Update P function
        P = P_update;
        pp = kron(ones(1, s1*s2*nb*nq), P);
    end
    
    toc 
    
    % Check convergence; if exitflag = 1 for all w, we are done 
    diff_feas = max(abs(exitflag-1));
    
    % If not, update the set of feasible w's
    w_feasible = w_feasible(exitflag==1);
    nfeas = length(w_feasible);
    
end

%% Display probabilities
for i = 1:length(w_feasible)
    iw = find(w_grid == w_feasible(i));
    xp=find(X_all(:,iw)>10^-4); %gives the indices of all elements of X > 10ˆ-4
    disp(['v =',num2str(w_grid(i))]);
    disp('y1 y2 Q b w prob')
    disp('———————————————————')
    disp([yy1(xp)', yy2(xp)', QQ(xp)', bb(xp)', ww(xp)', X_all(xp,iw)]);
end

%% Calculate transition probabilities
% for each promised utility value (this period)
w_trans = zeros(nw);
for i = 1:nw
    pi = X_all(:,i);
    % for each next period value
    for j = 1:nw
        index = (ww == w_grid(j));
        w_trans(i,j) = sum(pi(index));
    end
end

%% Use Pareto weight to find initial values
% Pareto weight
alpha = 0.5;
% Maximize social planner's expected utility at period 0
social_planner = alpha.*P + (1-alpha).*w_feasible;
[w0, w0_index] = max(social_planner);


%% Simulation
% number of periods
T = 100;
% matrices
y1_sim = zeros(1,T+1);
y2_sim = zeros(1,T+1);
b_sim = zeros(1,T+1);
w_sim = zeros(1,T+1);
w_index = zeros(1,T+1);
conscdf = zeros(s1*s2*nw*nb,T+1);
% beginning w
w_sim(1) = w0;
w_index(1) = 13;

% in each period
%for t = 1:T
for t = 1:T
    % probability vector is determined by last period's future promised
    % utility and this period's income realization
    w_index(t)
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
    [~, w_index(t+1)] = min(abs(w_grid-w_sim(t+1)));
    w_index(t+1)
end

%% Plot simulation
figure;
subplot(2,3,1);
plot(1:100,y1_sim(1:100));
title('Income - Agent 1');
subplot(2,3,2);
plot(1:100,y2_sim(1:100));
title('Income - Agent 2');
subplot(2,3,3);
c1_sim = y1_sim-b_sim;
plot(1:100,c1_sim(1:100));
title('Consumption - Agent 1');
subplot(2,3,4);
c2_sim = y2_sim+b_sim;
plot(1:100,c2_sim(1:100));
title('Consumption - Agent 2');
subplot(2,3,5);
plot(1:100,b_sim(1:100));
title('Transfer');
subplot(2,3,6);
plot(1:100,w_sim(1:100));
title('Continuation Value');

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


%}