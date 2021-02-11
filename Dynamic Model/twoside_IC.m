% Incentive compatible allocations under both sided hiding
% key elements: 1) consumption allocation 2) future expected utility

%% Parameters
% set seed
rng(4);

% Parameters
% Preference of private good versus public good
a1 = 0.5; a2 = 0.5;

% Pareto weight
alpha = 0.5;

% Time discounting
beta = 0.96;

% Cost of hiding
delta = 0.8;

% agent 2's income
y2 = 3;

%% Functional forms
% Income process is uniform on the interval [ymin,ymax].
ymin = 1; ymax = 5;

% to simplify, let's say there are only two states of income, with equal
% probability
inc1 = [ymin ymin; ymax ymax];
inc2 = [ymin ymax; ymin ymax];

% agent 2 chooses how much to transfer to agent 1 (can be negative)
% the most they can transfer is all of agent 2's income. the least they can
% transfer is taking all of agent 1's income.
m = 10;
transfer = linspace(-ymin,ymin,m);

% grid of promised utility values is bounded below by autarky value and
% above by perfect risk sharing
n = 10;
Vmin = log(ymin)./(1-beta);
Vmax = (0.25.*log(ymin) + 0.5.*log((ymin+ymax)./2) + 0.25.*log(ymax))./(1-beta);
utility_grid = linspace(Vmin,Vmax,n);

% U = agent 1, V = agent 2, transfer = agent 1 to agent 2

%% Following Wang (1995)

% starting matrix is the entire U/V gridpoint universe
feasible_result = ones(n);

% linear indices of the current universe that we will loop through
list_indices = find(feasible_result==1);

% store results
T = zeros(length(list_indices),4);
W = zeros(length(list_indices),4);
Z = zeros(length(list_indices),4);

% for each U (agent 1) / V (agent 2) pair
for pair = 1:length(list_indices)
    % use the matrix index for easy notation
    [i,j] = ind2sub(n, list_indices(pair));
    
    % we will keep looking for possible transfers and continuation
    % utilities until we find one that fulfills all the conditions, or
    % exhaust the search
    feasible = false;
    t11counter = 0;
    while feasible == false && t11counter<m+1
        
        % Vary transfers from agent 1 to agent 2
        % start from lowest income state
        for t11 = 1:m
            t11counter = t11counter+1;
            T_test(1,1) = transfer(t11);
            
            % when agent 1 report higher income, transfer to agent 2 must
            % be higher
            T_feasible_21 = linspace(T_test(1,1),inc1(2,1)-0.001,m);
            
            % when agent 2 report higher income, transfer to agent 2 must
            % be lower
            T_feasible_12 = linspace(-inc2(1,2)+0.001,T_test(1,1),m);
            
            for t21 = 1:m
                T_test(2,1) = T_feasible_21(t21);
                
                for t12 = 1:m
                    T_test(1,2) = T_feasible_12(t12);
                    
                    T_feasible_22 = linspace(T_test(1,2),T_test(2,1),m);
                    for t22 = 1:m
                        T_test(2,2) = T_feasible_22(t22);
                        
                        % loop through all gridpoints to find starting value of Z
                        % (continuation utility of agent 1)
                        for k = 1:n
                            % create an order such that it looks for
                            % continuation value near the objective
                            % first
                            % for even indices
                            if mod(k,2) == 0 && k<=min(2*i-2,2*n-2*i+1)
                                k_permute = i - k/2;
                            elseif mod(k,2) == 1 && k<=min(2*i-2,2*n-2*i+1)
                                k_permute = i + (k-1)/2;
                            elseif i>n/2
                                k_permute = n-k+1;
                            else
                                k_permute = k;
                            end
                            
                            % starting value of Z, Z(1,1) and Z(1,2)
                            Z_11 = utility_grid(k_permute);
                            
                            % calculate Z using IC constraints of agent 1
                            Z_21 = Z_11 + log(inc1(2,1) - T_test(1,1))./beta - log(inc1(2,1) - T_test(2,1))./beta;
                            Z_22 = utility_grid + log(inc1(2,2) - T_test(1,2))./beta - log(inc1(2,2) - T_test(2,2))./beta;
                            
                            % expected continuation utility
                            U_test = mean(mean(log(inc1-T_test))) + 0.25.*beta.*Z_11 + 0.25.*beta.*Z_21 + 0.25.*beta.*Z_22 + 0.25.*beta.*utility_grid;
                            
                            % is this feasible to choose Z so that we arrive at U? (promise keeping constraint)
                            feasible_U = (min(abs(U_test - utility_grid(i)))<0.01);
                            
                            % continue if feasible. If not, find another starting value
                            % of Z
                            if feasible_U == true
                                
                                [~, index] = min(abs(U_test - utility_grid(i)));
                                Z_12 = utility_grid(index);
                                Z_22 = Z_22(index);
                                
                                % loop through all gridpoints to find starting value of
                                % W (agent 2)
                                for l = 1:n
                                    % create an order such that it looks for
                                    % continuation value near the objective
                                    % first
                                    % for even indices
                                    if mod(l,2) == 0 && l<=min(2*j-2,2*n-2*j+1)
                                        l_permute = j - l/2;
                                    elseif mod(l,2) == 1 && l<=min(2*j-2,2*n-2*j+1)
                                        l_permute = j + (l-1)/2;
                                    elseif i>n/2
                                        l_permute = n-l+1;
                                    else
                                        l_permute = l;
                                    end
                                    % starting value of W, W(1,1) and W(2,1)
                                    W_11 = utility_grid(l_permute);
                                    
                                    % calculate W using IC constraints of agent 2
                                    W_12 = W_11 + log(inc2(1,2) + T_test(1,1))./beta - log(inc2(1,2) + T_test(1,2))./beta;
                                    W_22 = utility_grid + log(inc2(2,2) + T_test(2,1))./beta - log(inc2(2,2) + T_test(2,2))./beta;
                                    
                                    % expected continuation utility
                                    V_test = mean(mean(log(inc2+T_test))) + 0.25.*beta.*W_11 + 0.25.*beta.*W_12 + 0.25.*beta.*W_22 + 0.25.*beta.*utility_grid;
                                    
                                    % is this feasible to choose W so that we arrive at V? (promise keeping constraint)
                                    feasible_V = (min(abs(V_test - utility_grid(j)))<0.01);
                                    
                                    if feasible_V==true
                                        [~, index2] = min(abs(V_test - utility_grid(j)));
                                        W_21 = utility_grid(index2);
                                        W_22 = W_22(index2);
                                    end
                                    
                                    % if a feasible allocation is found, end the loop
                                    feasible = feasible_U && feasible_V;
                                    
                                    if feasible == true
                                        break
                                    end
                                    
                                end
                            end
                            
                            if feasible == true
                                break
                            end
                        end
                    
                        if feasible == true
                            break
                        end
                    end
                    
                    if feasible == true
                        break
                    end
                end
                
                if feasible == true
                    break
                end    
            end
            
            if feasible == true
                break
            end    
        end
    end
    
    % save feasible result
    feasible_result(i,j) = feasible;
    if feasible == true
        T(pair,:) = [T_test(1,:) T_test(2,:)];
        W(pair,:) = [W_11 W_12 W_21 W_22];
        Z(pair,:) = [Z_11 Z_12 Z_21 Z_22];
    end
end
