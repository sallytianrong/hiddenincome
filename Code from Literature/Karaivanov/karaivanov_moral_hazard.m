%% Computing Moral Hazard Programs With Lotteries Using Matlab (Karaivanov)
% this code file copied the verbatim code from Karaivanov's manuscript.

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part I)
clear all %clears the memory from all variables
%1. Assign values to the parameters
g=.5; %gamma
k=1; %kappa
d=.5; %delta
a=.7; %alpha
U=1; %reservation utility
%2. Define the grids
%Consumption Grid
nc=20; %number of points in the consumption grid
cmin=10.^-8; %lowest possible consumption level (can’t be 0 for the chosen function)
cmax=3; %highest possible consumption level
c=linspace(cmin,cmax,nc); %creates the actual grid for consumption as 20
%equally spaced values between cmin and cmax
%Action level
nz=10; %number of grid points
zmin=10.^-8; %minimum effort level
zmax=1-10.^-8; %maximum effort level
z=linspace(zmin,zmax,nz); %creating the grid
%Output
nq=2; %number of grid points
ql=1; %low output level
qh=3; %high output level
q=[ql,qh]; %two possible output values by assumption, in order to
%simplify the computation of probabilities

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part 2)
P(1:2:nz*nq-1)=1-z.^a; %the conditional probabilities corresponding to ql
P(2:2:nz*nq)=z.^a; %the conditional probabilities corresponding to qh

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part 3)
UB=ones(nz*nq*nc,1); %the vector of upper bounds
LB=zeros(nz*nq*nc,1); %lower bounds

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part 4)
f = -kron(ones(1, nz), kron(q, ones(1, nc))) + kron(ones(1, nq*nz), c);

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part 5)
Aeq1=ones(1, nz*nq*nc); %the coefficients are ones on each π
beq1=1; %the sum of probabilities needs to be 1.

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part 6)
Aeq2 = kron(eye(nz*nq), ones(1,nc)) - kron(kron(eye(nz), ones(nq,1)).*...
(P'*ones(1,nz)), ones(1,nq*nc));
beq2 = zeros(nz*nq, 1);

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part 7)
A1 = -(1-g)^-1*kron(ones(1, nq*nz),c).^(1-g) - k/(1-d)*...
(kron(1-z, ones(1,nc*nq))).^(1-d);
b1 = -U;

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part 8)
for iz=1: nz %loop on the recommended action level, z
zh = [1:iz-1 iz+1:nz]; %vector of all possible alternative action levels, ˆz
for jz=1:nz-1 %loop on the alternative action level, ˆz
%Constructing the constraints one by one
A2((nz-1)*(iz-1)+jz, :) = kron([zeros(1, iz-1), 1, zeros(1, nz-iz)],...
kron(ones(1,nq), A1(nc*nq*(iz-1)+1:nc*nq*(iz-1)+nc))) ...
+ kron([zeros(1, iz-1), 1, zeros(1,nz-iz)], ones(1, nq*nc)).*...
(kron(ones(1, nz), kron([P(2*zh(jz)-1)/P(2*iz-1), P(2*zh(jz))/P(2*iz)] ...
,-A1(nc*nq*(zh(jz)-1)+1: nc*nq*(zh(jz)-1)+nc))));
end
end
b2=zeros(nz*(nz-1), 1);

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part 9)
Aeq=[Aeq1; Aeq2]; %matrix of coefficients on the equality constraints
beq=[beq1; beq2]; %intercepts
A=[A1; A2]; %matrix of coefficients on the inequality constraints
b=[b1; b2]; %intercepts

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part 10a)
[X, vobj] = linprog(f, A, b, Aeq, beq, LB, UB);

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part 10b)
options = optimset('Display','off','TolFun', 10^-9,'MaxIter',150,'TolX', 10^-8);
[X, vobj] = linprog(f, A, b, Aeq, beq, LB, UB,[],options);

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part 11)
cc = kron(ones(1, nq*nz), c);
qq = kron(ones(1, nz), kron(q, ones(1, nc)));
zz = kron(z,ones(1, nc*nq));

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part 12)
xp=find(X >10^-4); %gives the indices of all elements of X > 10ˆ-4

%SAMPLE MATLAB CODE FOR MORAL HAZARD PROBLEMS (part 13)
disp('z q c prob')
disp('———————————————————')
disp([zz(xp)', qq(xp)', cc(xp)', X(xp)]);