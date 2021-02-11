%% Sargent-Ljungqvist Exercise 20.4 Thomas-Worrall meet Phelan-Townsend
% Following parameters from 20.4 and using code from Karaivanov
% One risk-neutral principal and one risk-averse agent. State variables are
% the transfer the agent receives this period and promised future utility.

%% Set up and parameters
clear all; close all; clc;

beta = 0.94;
a = 5;
gamma = 3;

% grid for Y (income), B (transfer) and W (continuation value)
% income grid
s = 5;
prob_y = ones(1,s).*s.^(-1);
y_min = 6;
y_max = 15;
y_grid = linspace(y_min,y_max,s);

% transfer grid
nb = 25;
b_min = (1-y_max+0.33);
b_max = y_max-y_min;
b_neg = linspace(b_min,0,(nb+1)/2);
b_grid = [b_neg(1:end-1) linspace(0,b_max,(nb+1)/2)];

% continuation value grid
nw = 25;
w_min = (y_min-a).^(1-gamma)./((1-beta).*(1-gamma));
w_max = w_min./20;
w_grid = linspace(w_min,w_max,nw);

% utility function
u = @(x) (x-a).^(1-gamma)./(1-gamma);

%% Iteration on feasible w-space, then on P-function

% Space of feasible w gridpoints, starting with the entire w-grid
exitflag = ones(1, nw);
w_feasible = w_grid(exitflag==1);
nfeas = length(w_feasible);

% initial guess of P function
P = -2.*w_feasible;
P_update = P;
pp = kron(ones(1, s*nb), P);

% set up for convergence
diff_feas = 100;
error = 0.01;

% convergence parameters
diff = 100;
error = 0.01;

%% outer loop: iterate on the feasible space of w gridpoints
while diff_feas>error
    
    % vector to store optimization flags
    exitflag = ones(1,nfeas);
    
    % A matrix to store probabilities
    X_all = zeros(nfeas*nb*s,nfeas);
    
    %% Constraints set up
    % kronecker product set up
    yy = kron(y_grid, ones(1, nb*nfeas));
    bb = kron(ones(1, s), kron(b_grid, ones(1, nfeas)));
    ww = kron(ones(1, s*nb), w_feasible);
    
    % Upper bounds and lower bounds (between 0 and 1)
    UB = ones(s*nb*nfeas,1); %the vector of upper bounds
    LB = zeros(s*nb*nfeas,1); %lower bounds
    % not allowing consumption to be negative
    neg = (yy+bb-a<=0);
    UB(neg) = 0;
    
    % sum of probabilities (5)
    Aeq1 = ones(1, s*nb*nfeas); %the coefficients are ones on each π
    beq1 = 1; %the sum of probabilities needs to be 1.
    
    % incentive compatibility (3)
    counter = 0;
    Aineq = zeros(2*(s-1),s*nb*nfeas);
    
    % All IC constraints
    for i = 1:s % each state of the world
        yi_index = (yy == y_grid(i));
        alt = [1:i-1 i+1:s]; % all other states of the world
        for j = 1:s-1 % each alternate state of the world
            counter = counter+1;
            yj_index = (yy == y_grid(alt(j)));
            Aineq(counter,yi_index) = -(u(yy(yi_index)+bb(yi_index))+ beta.*ww(yi_index))./prob_y(i);
            Aineq(counter,yj_index) = (u(yy(yj_index)+bb(yj_index))+ beta.*ww(yj_index))./prob_y(alt(j));
            ic_neg = false(1,s*nb*nfeas);
            ic_neg(yj_index) = (y_grid(i)+bb(yi_index)<=0);
            Aineq(counter,neg) = 0;
            Aineq(counter,ic_neg) = 0;
        end
    end
    bineq = zeros(s*(s-1),1);
    
    % conditional probabilities add up to unconditional probabilities of y
    Aeq3 = kron(eye(s),ones(1,nfeas*nb));
    beq3 = prob_y';
    
    %% inner loop: iterate on P function until convergence
    diff = 100;
    
    while diff>error
        % for each v of continuation value
        for k = 1:nfeas
            
            v = w_grid(k);
            
            % promise keeping constraint (2)
            %Aeq_U = (kron(y_grid,ones(1,nb)) + kron(ones(1,s),b_grid) - a).^(1-gamma)./(1-gamma);
            %Aeq2 = kron(Aeq_U,ones(1,nw)) + kron(ones(1,s*nb),beta.*w_grid);
            Aeq2 = (yy + bb - a).^(1-gamma)./(1-gamma) + beta.*ww;
            beq2 = v;
            
            % all constraints
            Aeq=[Aeq1; Aeq2; Aeq3]; %matrix of coefficients on the equality constraints
            beq=[beq1; beq2; beq3]; %intercepts
            
            % Objective function (1)
            f = bb - beta.*pp;
            
            % Use linprog to optimize
            options = optimset('Display','off','TolFun', 10^-3,'MaxIter',750,'TolX', 10^-3);
            [X, fval, exitflag(k)] = linprog(f, Aineq, bineq, Aeq, beq, LB, UB,[],options);
            
            % if optimization was successful, update value function
            if exitflag(k) == 1
                % save X matrix
                X_all(:,k) = X;
                
                % update P
                P_update(k) = -fval;
            end
        end
        
        % check convergence of value function;
        diff = max(abs(P-P_update))
        
        % update value function
        P = P_update;
        
    end
    
    %% Check feasibility
    % optimization was feasible
    exitflag_feasible = (exitflag==1);
    
    % Check convergence of w-space
    diff_feas = max(abs(exitflag_feasible-1));
    
    % Update w-space
    w_feasible = w_feasible(exitflag_feasible);
    nfeas = sum(exitflag_feasible)
    
    % Update initial guess of P function
    P = P(exitflag_feasible);
    P_update = P;
    pp = kron(ones(1, s*nb), P);
    
end

%% Display probabilities
% for each v
for i = 1:nfeas
    % display only the positive probabilities
    xp=find(X_all(:,i)>10^-4);
    % display v, income, transfer, continuation value and probability in this order
    disp(['v =',num2str(w_grid(i))]);
    disp('y b w prob')
    disp('———————————————————')
    disp([yy(xp)', bb(xp)', ww(xp)', X_all(xp,i)]);
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

if isempty(eig1_index) == true
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
    title('Invariant Distribution of agent utility')
    subplot(1,2,2);
    plot(prob_p(:,1), prob_p(:,2));
    title('Invariant Distribution of p')
end

% plot P function
figure;
plot(w_feasible,P);
title('Profits as a Function of Agent Utility');

%% Simulation
% number of periods
T = 100;
% matrices
y_sim = zeros(1,T+1);
b_sim = zeros(1,T+1);
w_sim = zeros(1,T+1);
w_index = zeros(1,T+1);
conscdf = zeros(s*nfeas*nb,T+1);
% beginning w
w0 = -1;
[~, w_index(1)] = min(abs(w_feasible-w0));
w_sim(1) = w_feasible(w_index(1));

% in each period
for t = 1:T
    % probability vector is determined by last period's future promised
    % utility and this period's income realization
    pi = X_all(:,w_index(t));
    picum = cumsum(pi);
    % calculate probability of consumption
    [cons, cons_index] = sort(yy+bb);
    pi_cons = pi(cons_index);
    conscdf(:,t) = cumsum(pi_cons);
    % realization of income, transfer and continuation value
    randnum = rand(1); % random number
    crit1 = (picum > randnum); % random number is less than cumulative distribution
    crit2 = (pi~=0); % points that have mass
    % values that fit criterion
    w_next = ww(crit1 & crit2);
    y_all = yy(crit1 & crit2);
    b_all = bb(crit1 & crit2);
    % update values
    w_sim(t+1) = w_next(1);
    y_sim(t) = y_all(1);
    b_sim(t) = b_all(1);
    % index
    [~, w_index(t+1)] = min(abs(w_grid-w_sim(t+1)));
end

%% Plot simulation
figure;
subplot(2,2,1);
plot(1:T,y_sim(1:T));
title('Income');
subplot(2,2,2);
plot(1:T,b_sim(1:T));
title('Transfer');
subplot(2,2,3);
c_sim = y_sim+b_sim;
plot(1:T,c_sim(1:T));
title('Consumption');
subplot(2,2,4);
plot(1:T,w_sim(1:T));
title('Continuation Value');

%% Plot distribution of consumption
figure;
plot(cons,conscdf(:,5),cons,conscdf(:,50),cons,conscdf(:,80),cons,conscdf(:,100))
legend('t=5','t=50','t=80','t=100');
xlim([a y_max]);
ylim([0 1]);