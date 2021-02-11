%%% Jakiela and Ozier Model Simulation %%%
% Questions: 1. is hiding an equilibrium in the model? 2. Is this model a
% good model of hiding? 3. What is missing in this model?

%% What's the optimal investment given risk aversion, without random utility
% term?

% risk aversion parameter
n = 100;
rho = linspace(0,0.9999,n);
% endowment (small=80, large=180)
m = 180;
% optimal investment is a fraction of the endowment
b = min((1-4.^(-1./rho))./(4.^(1-1./rho)+1),1).*m;
% plot
plot(rho,b);

%% Adding random utility term
% noise parameter
epsilon = 0.5;
% possible investment levels are 0 to m (column vector)
levels = (0:m)';
% calculate utility for each investment level
utility = 0.5.*((m-levels).^(1-rho)./(1-rho)) + 0.5.*((m+4.*levels).^(1-rho)./(1-rho));
% calculate a sum
sum_u = sum(exp(utility./epsilon));
% calculate expected investment
expected_inv = sum(levels.*exp(utility./epsilon))./sum_u;

%% Random utility term with tax
% tax
t = 0.045;
% calculate utility for each investment level
utility_tax = 0.5.*(((1-t).*(m-levels)).^(1-rho)./(1-rho)) + 0.5.*(((1-t).*(m+4.*levels)).^(1-rho)./(1-rho));
% calculate a sum
sum_u_tax = sum(exp(utility_tax./epsilon));
% calculate expected investment
expected_inv_tax = sum(levels.*exp(utility_tax./epsilon))./sum_u_tax;
% plot differences with and without random utility term
plot(rho,expected_inv,rho,expected_inv_tax,rho,b);
legend('Random Utility','Random Utility with Tax','No Random Utility','Location','Southwest');
% note: difference between the random utility investment levels increase with tax

%% Large endowment - no random utility
% strategy 1: reveal and pay the porportional tax
% b = optimal strategy
u_public = 0.5.*(((1-t).*(m-b)).^(1-rho)./(1-rho)) + 0.5.*(((1-t).*(m+4.*b)).^(1-rho)./(1-rho));
% strategy 2: pretend to get small endowment (investing 80 or less)
b_pretend = min(b, 80);
u_pretend = 0.5.*(((1-t).*(m-b_pretend-100)+100).^(1-rho)./(1-rho)) + 0.5.*(((1-t).*(m+4.*b_pretend-100)+100).^(1-rho)./(1-rho));
u_optimal = max(u_public, u_pretend);
plot(rho, u_public, rho, u_pretend);
legend('No Hiding Motive','Hiding Motive')
% graph optimal investment
b_optimal = b;
b_optimal(u_pretend>u_public) = b_pretend(u_pretend>u_public);
plot(rho, b, rho, b_optimal);
legend('No Hiding Motive','Hiding Motive')

%% Large endowment - random utility
% fix rho
rho_fixed=0.75;
% calculate utility for each investment level
utility_public_random = 0.5.*(((1-t).*(m-levels)).^(1-rho_fixed)./(1-rho_fixed)) + 0.5.*(((1-t).*(m+4.*levels)).^(1-rho_fixed)./(1-rho_fixed));
% calculate a sum
sum_u_public = sum(exp(utility_public_random./epsilon));
% calculate probability of investment
prob_public = exp(utility_public_random./epsilon)./sum_u_public;
% calculate utility for each investment level
smallm = (levels<=80);
utility_pretend_random = 0.5.*(((1-t).*(m-levels)+smallm.*t.*100).^(1-rho_fixed)./(1-rho_fixed)) + 0.5.*(((1-t).*(m+4.*levels)+smallm.*t.*100).^(1-rho_fixed)./(1-rho_fixed));
% calculate a sum
sum_u_pretend = sum(exp(utility_pretend_random./epsilon));
% calculate probability of investment
prob_pretend = exp(utility_pretend_random./epsilon)./sum_u_pretend;
% plot probability of investment at every level
plot(levels,prob_public,levels,prob_pretend)
legend('No Hiding Motive','Hiding Motive')

%% A continuous hiding strategy: can claim endowment is equal to investment
b_cont = (1-(4-5.*t).^(-1./rho)).*m./((4-5.*t).^(1-1./rho)+1);
plot(rho, b, rho, b_optimal, rho, b_cont)
legend('No Hiding Motive','Hiding Motive','Continuous Hiding')

% calculate utility for each investment level
utility_cont_random = 0.5.*(m-levels).^(1-rho_fixed)./(1-rho_fixed) + 0.5.*((1-t).*5.*levels + m - levels).^(1-rho_fixed)./(1-rho_fixed);
% calculate a sum
sum_u_cont = sum(exp(utility_cont_random./epsilon));
% calculate probability of investment
prob_cont = exp(utility_cont_random./epsilon)./sum_u_cont;
% plot probability of investment at every level
plot(levels,prob_public,levels,prob_pretend,levels,prob_cont)
legend('No Hiding Motive','Hiding Motive','Continuous Hiding')

%% Willingness to pay to hide everything (no taxes paid) before investment
% realization
wtp_pre = zeros(1,n);
for i = 1:n
    wtp_fun = @(x) 0.5.*((m-x-(m-x).*(1-4.^(-1./rho(i)))./(4.^(1-1./rho(i))+1)).^(1-rho(i))./(1-rho(i)))...
        + 0.5.*((m-x+4.*(m-x).*(1-4.^(-1./rho(i)))./(4.^(1-1./rho(i))+1)).^(1-rho(i))./(1-rho(i))) - u_optimal(i);
    wtp_pre(i) = fzero(wtp_fun, 50);
end
% Paying before investing changes optimal investments
b_exante = (m-wtp_pre).*(1-4.^(-1./rho))./(4.^(1-1./rho)+1);

% Willingness to pay to hide everything (no taxes paid) after investment
% realization equals the tax
hide_ind = (b_optimal<=80);
wtp_lost = hide_ind.*t.*(80-b_optimal) + (1-hide_ind).*t.*(m-b_optimal);
wtp_won = hide_ind.*t.*(80+4.*b_optimal) + (1-hide_ind).*t.*(m+4.*b_optimal);

figure;
subplot(1,2,1);
plot(rho, wtp_pre, rho, wtp_lost, rho, wtp_won)
legend('Before Realization','Lost Investment','Won Investment')
title('WTP')
subplot(1,2,2);
plot(rho,b_exante,rho,b_optimal);
legend('Ex Ante','Ex Post')
title('Optimal Investment')

%% Three strategies in price treatment
% list of prices
%p = [10;20;30;40;50;60];
p = (1:60)';
% strategy 1: never pay, utility is same as public. optimal investment is
% same as public.
u_never = repmat(u_optimal,length(p),1);

% strategy 2: always pay
b_always = (1-4.^(-1./rho)).*(m-p)./(4.^(1-1./rho)+1);
u_always = 0.5.*(m-b_always-p).^(1-rho)./(1-rho) + 0.5.*(m+4.*b_always-p).^(1-rho)./(1-rho);

% strategy 3: only pay when investment is successful
b_heads = ((1-4.^(-1./rho)-t).*m + 4.^(-1./rho).*p)./(4.^(1-1./rho)+1-t);
for i = 1:length(p)
    for j = 1:length(rho)
        ind = (b_heads(i,j) + t.*100./(4.^(1-1./rho(j))+1-t) < 80);
        b_heads(i,j) = b_heads(i,j) + ind.*t.*100./(4.^(1-1./rho(j))+1-t);
    end
end
index = (b_heads<80);
u_heads = 0.5.*((1-t).*(m-b_heads)+t.*index.*100).^(1-rho)./(1-rho) + 0.5.*(m+4.*b_heads-p).^(1-rho)./(1-rho);

% optimal strategy
strategy = zeros(size(u_never));
strategy(u_never>max(u_always,u_heads))=1;
strategy(u_always>max(u_never,u_heads))=2;
strategy(u_heads>max(u_never,u_always))=3;

figure;
h = heatmap(strategy);
h.XLabel = 'Risk Aversion';
h.YLabel = 'Price';
h.GridVisible = 'off';


%%
% optimal investment and utility
investment = zeros(size(u_never));
b_optimal_rep = repmat(b_optimal,length(p),1);
investment(strategy==1) = b_optimal_rep(strategy==1);
investment(strategy==2) = b_always(strategy==2);
investment(strategy==3) = b_heads(strategy==3);

% figures
figure;
subplot(2,3,1);
plot(rho,u_never(1,:),rho,u_always(1,:),rho,u_heads(1,:));
title('p=10');
legend('Never Pay','Always Pay','Pay when successful');

subplot(2,3,2);
plot(rho,u_never(2,:),rho,u_always(2,:),rho,u_heads(2,:));
title('p=20');
legend('Never Pay','Always Pay','Pay when successful');

subplot(2,3,3);
plot(rho,u_never(3,:),rho,u_always(3,:),rho,u_heads(3,:));
title('p=30');
legend('Never Pay','Always Pay','Pay when successful');

subplot(2,3,4);
plot(rho,u_never(4,:),rho,u_always(4,:),rho,u_heads(4,:));
title('p=40');
legend('Never Pay','Always Pay','Pay when successful');

subplot(2,3,5);
plot(rho,u_never(5,:),rho,u_always(5,:),rho,u_heads(5,:));
title('p=50');
legend('Never Pay','Always Pay','Pay when successful');

subplot(2,3,6);
plot(rho,u_never(6,:),rho,u_always(6,:),rho,u_heads(6,:));
legend('Never Pay','Always Pay','Pay when successful');
title('p=60');

% figures
figure;
subplot(2,3,1);
plot(rho,b,rho,b_optimal,rho,investment(10,:));
title('p=10');
legend('Private Treatment','Public treatment','Price treatment');

subplot(2,3,2);
plot(rho,b,rho,b_optimal,rho,investment(20,:));
title('p=20');
legend('Private Treatment','Public treatment','Price treatment');

subplot(2,3,3);
plot(rho,b,rho,b_optimal,rho,investment(30,:));
title('p=30');
legend('Private Treatment','Public treatment','Price treatment');

subplot(2,3,4);
plot(rho,b,rho,b_optimal,rho,investment(40,:));
title('p=40');
legend('Private Treatment','Public treatment','Price treatment');

subplot(2,3,5);
plot(rho,b,rho,b_optimal,rho,investment(50,:));
title('p=50');
legend('Private Treatment','Public treatment','Price treatment');

subplot(2,3,6);
plot(rho,b,rho,b_optimal,rho,investment(60,:));
legend('Private Treatment','Public treatment','Price treatment');
title('p=60');