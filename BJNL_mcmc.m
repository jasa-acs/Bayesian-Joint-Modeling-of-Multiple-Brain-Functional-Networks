function [ output_args ] = BJNL_mcmc( YY2, SS, N, p, membership, n, nmc,...
    burnin, isScriptVersion, outputFolder, prefix, nsub )

% Function implementing the MCMC for BJNL. Called from the main.m function.

z = membership;

% Preallocate storage for iteration results
C_save = zeros(p,p*N,nmc);
Sig_save = zeros(p,p*N,nmc);
Edge_save = zeros(p,p*N,nmc);
edgeprob_save = zeros(N,p*(p-1)/2,nmc);
Etaout = zeros(p*(p-1)/2,nmc);
Intout = zeros(p*(p-1)/2,nmc);
sigma_save = zeros(N,N,nmc);
zeta_save = zeros(nmc,2);
phiout = zeros(p*(p-1)/2, nmc);

% Storage for trace plots
C_save_all = zeros(p,p*N,burnin+nmc);
Sig_save_all = zeros(p,p*N,burnin+nmc);
tau_save_all = zeros(p,p*N,burnin+nmc);

% get the count of off diagonal elements.
edgeCount = p*(p-1)/2;

lambda0 = 100; lambda1 = 0.1;
Edge_arr = zeros(p,p,N);
C_arr = zeros(p,p,N);
cArrPrevIter = zeros(p,p,N);
Sig_arr = zeros(p,p,N);
tau_arr = zeros(p,p,N);

gamma = zeros(p,nsub,N);
Lambda_arr = zeros(p,p,N);

% Populate with starting values
for nn=1:N
    C_arr(:,:,nn) = inv(reshape(SS(:,(nn*p-p+1):nn*p),[p,p]) + 0.5*eye(p));
    Edge_arr(:,:,nn) = (abs(reshape(C_arr(:,:,nn),[p,p]))>0.005);
    Sig_arr(:,:,nn) = reshape(SS(:,(nn*p-p+1):nn*p),[p,p]) + 0.5*eye(p);
    Lambda_arr(:,:,nn) = lambda1*Edge_arr(:,:,nn) + lambda0*(1-Edge_arr(:,:,nn));
end

prob = zeros(N,p*(p-1)/2);
alpha = 0.1;

% Hyperparameters, settings
at = 0.1; bt = 1;
az = 0.1; bz = 1;
as = 0.1; bs = 1;
api = 0.1; bpi = 1;
pi_inc = 0.1;
theta = 0.1*ones(N,1);
s = 1e-2;
t =1e-6;
agam = 10000000; bgam = 10;
t_df = 7.3;
sigma2_tilde = sqrt(3.14*3.14*(t_df-2)/(3*t_df));
sigma_u = sqrt(sigma2_tilde)*ones(p*(p-1)/2,1);
phi = ones(p*(p-1)/2,1);
aa = 0.1;

% Initialize values
Eta = zeros(edgeCount,1); % initial atom values
fold = ones(edgeCount,1);
inter = normrnd(0,1,[p*(p-1)/2,1]);
inter_p = normrnd(0,1,[p*(p-1)/2,1]);
nus = betarnd(1,aa,[1 edgeCount]);             % stick-breaking random variables
nu = nus.*cumprod([1 1-nus(:,1:edgeCount-1)]); % category probabilities
kjs = 5;
ph = unidrnd(kjs,[p*(p-1)/2 1]);            % initial atoms allocation index
u = unifrnd(0,nu(ph)');
zeta_int = 1;
zeta_intp = 1;
Sigma = 1;
u_star = normrnd(0,1,[p*(p-1)/2,N]);

% start the MCMC
for iter = 1: burnin+nmc
    
    % Print the current iteration
    if(mod(iter,100)==0)
        fprintf('iter = %d, \n',iter);
    end
    
    SSS = zeros(p,N*p);
    for nn=1:N
        % Get each subject level sum of squares
        S = zeros(p,p);
        % Handle experimental condition 1
        if(z(nn)==1)
            for sub=1:nsub
                S = S + ( reshape(YY2(:,1:n(nn),sub,nn),[p,n(nn)]) -...
                    gamma(:,sub,1)*ones(1,n(nn))) * ...
                    (reshape(YY2(:,1:n(nn),sub,nn),[p,n(nn)]) -...
                    gamma(:,sub,1)*ones(1,n(nn)))';
            end
        end
        % Handle experimental condition 0
        if(z(nn)==0)
            for sub=1:nsub
                S = S + (reshape(YY2(:,1:n(nn),sub,nn),[p,n(nn)]) -...
                    gamma(:,sub,2)*ones(1,n(nn)))*...
                    (reshape(YY2(:,1:n(nn),sub,nn),[p,n(nn)]) -...
                    gamma(:,sub,2)*ones(1,n(nn)))';
            end
        end
        
        SSS(:,(nn*p - p+1):nn*p) = S;
    end
    
    % Loop over each experimental condition. Loop can be parallelized.
    for nn = 1:N
        
        % Grab the corresponding sum of squares for exp condition nn.
        S = SSS(:,(nn*p - p+1):nn*p);
        % Get the edge set
        Edge = reshape(Edge_arr(:,:,nn),p,p);
        % Get the precision
        C = reshape(C_arr(:,:,nn),p,p);
        % Covariance
        Sig = reshape(Sig_arr(:,:,nn),p,p);
        
        % If first iteration, get the covariance by inverting the precision
        % matrix
        if(iter==1)
            Sig = inv(C);
        end
        
        tau = zeros(p,p);
        
        tt=0;
        % sample the Precision, Covariance, and Edge set for the nn-th experimental condition
        
        % Loop over each node
        for i = 1:p
            
            % ind is the indices of the edges less than the current node
            % (lower triangle of the matrix)
            ind = 1:p; ind(i) = [];
            % Obtain the corresponding elements of the covariance matrix
            Sig11 = Sig(ind,ind); Sig12 = Sig(ind,i);
            % Get estimate for precision matrx
            invC11 = Sig11 - Sig12*Sig12'/Sig(i,i);
            % Ensure the estimate is positive definite
            invC11 = (invC11 + invC11')/2 + 0.000001*eye(p-1);
            C11 = C(ind,ind);
            
            % make sure not on final node
            if(i<p)
                % Loop over all edges (upper triangle)
                for j=(i+1):p
                    tt = tt + 1;
                    % Calculate edge weight
                    theta_nn = 1-normcdf(0, inter_p(tt)*z(nn) +  Eta(ph(tt)),sigma_u(tt));
                    
                    % Parameters  (see Wang 2012)
                    Cadjust = max(abs(C(i,j)),10^-6);
                    mu_prime0 = min(sqrt(lambda0^2)./Cadjust,10^12);
                    tau_temp0 =  1./rand_ig(mu_prime0,lambda0^2);
                    beta = C(i,ind);
                    jj = j-1; ind1 = 1:(p-1); ind1(ind1==jj) = [];
                    sigma20 = 1/((S(i,i) + alpha)*invC11(jj,jj) + 1/tau_temp0);
                    mu0 = -(sigma20)*(invC11(jj,ind1)*beta(ind1)' + S(i,j));
                    tau_temp1 = 1/gamrnd(at + 1/2, 1/(bt + 0.5*C(i,j)^2));
                    beta = C(i,ind);
                    jj = j-1; ind1 = 1:(p-1); ind1(ind1==jj) = [];
                    sigma21 = 1/((S(i,i) + alpha)*invC11(jj,jj) + 1/tau_temp1);
                    mu1 = -(sigma21)*(invC11(jj,ind1)*beta(ind1)' + S(i,j));

                    % Consider two cases: 
                    %   Case 0: No edge (spike)
                    %   Case 1: Edge (slab)
                    % Calculate the log likelihood under each case
                    lprob0 = log(lambda0^2/2) - (0.5*lambda0^2*tau_temp0) +...
                        log(1/sqrt(tau_temp0)) + (0.5*mu0^2/sigma20) +...
                        log(sqrt(sigma20));
                    lprob1 = log(gampdf(1/tau_temp1,at,1/bt)) +...
                        log(1/sqrt(tau_temp1)) + (0.5*mu1^2/sigma21) +...
                        log(sqrt(sigma21));
                    
                    % Make sure the log likelihoods dont cause numerical
                    % problems
                    if (lprob0 < -100)
                        lprob0 = -100;
                    end
                    if (lprob1 < -100)
                        lprob1 = -100;
                    end
                    if (lprob0 > 100)
                        lprob0 = 100;
                    end
                    if (lprob1 > 100)
                        lprob1 = 100;
                    end
                    
                    % Calculate the probability of an edge (pstar)
                    mprob = max(abs(lprob0), abs(lprob1));
                    pstar = ( theta_nn * exp(lprob1 + mprob) )/( theta_nn*exp(lprob1+mprob) + ((1-theta_nn)) * exp(lprob0+mprob) );
                    
                    % Random draw for edge with probability pstar
                    Edge(i,j) = binornd(1, pstar);
                    Edge(j,i) = Edge(i,j);
                    
                    % Update step if there is no edge
                    if (Edge(i,j)==0)
                        Cadjust = max(abs(C(i,j)),10^-6);
                        lambda_prime = lambda0^2;
                        mu_prime = min(sqrt(lambda_prime)./Cadjust,10^12);
                        tau_temp =  1./rand_ig(mu_prime,lambda_prime);
                    % Update step if there is an edge  
                    else
                        tau_temp = 1/gamrnd(at + 1/2, 1/(bt + 0.5*C(i,j)^2));
                    end
                    % Store the tau parameter
                    tau(i,j) = tau_temp; tau(j,i) = tau_temp;
                    
                    % Reparameterize following Wang 2012
                    beta = C(i,ind);
                    jj = j-1; ind1 = 1:(p-1); ind1(ind1==jj) = [];
                    sigma2 = 1/((S(i,i) + alpha)*invC11(jj,jj) + 1/tau_temp);
                    mu = -(sigma2)*(invC11(jj,ind1)*beta(ind1)' + S(i,j));
                    C(i,j) = normrnd(mu,sqrt(sigma2)); C(j,i) = C(i,j);
                    
                end
            end
            
            % Update precision matrix
            ind = 1:p; ind(i) = [];
            Ci = (S(i,i)+alpha)*invC11+diag(1./tau(i,ind));
            Ci_chol = chol(Ci);
            mu_i = -Ci\S(ind,i);
            beta = mu_i+ Ci_chol\randn(p-1,1);       
            gam = gamrnd(nsub*n(nn)/2+1,(S(i,i)+ alpha)\2);
            C(ind,i) = beta;
            C(i,ind) = beta;
            C(i,i) = gam+beta'*invC11*beta;
            
            
            % Update Covariance matrix according to one-column change of precision matrix
            invC11beta = invC11*beta;
            Sig(ind,ind) = invC11+invC11beta*invC11beta'/gam;
            Sig12 = -invC11beta/gam;
            Sig(ind,i) = Sig12;
            Sig(i,ind) = Sig12';
            Sig(i,i) = 1/gam;
        end % end of i loop
        
        % Store the estimates for edge set, precision, covariance, and tau
        Edge_arr(:,:,nn) = Edge;
        C_arr(:,:,nn) = C;
        Sig_arr(:,:,nn) = Sig;
        tau_arr(:,:,nn) = tau;
        
    end 
    
    % For each possible edge in each experimental condition, store the
    % probability of an edge (w in the paper)
    for nn = 1:N
        tt=0;
        for i = 1:p
            for j = 1:p
                if(i>j)
                    tt = tt + 1;
                    theta_nn = 1-normcdf(0, inter_p(tt)*z(nn) +  Eta(ph(tt)),sigma_u(tt));
                    prob(nn,tt) = theta_nn;
                end
            end
        end
    end
    
     L0 = -10000; U0 = 0; 
    L1 = 0; U1 = 10000;
    for tt=1:p*(p-1)/2
    u_star(tt,z==0) = Eta(ph(tt)) + sigma_u(tt)*trandn((L0-Eta(ph(tt)))/sigma_u(tt),(U0-Eta(ph(tt)))/sigma_u(tt));
    u_star(tt,z==1) = Eta(ph(tt)) + inter_p(tt) + sigma_u(tt)*trandn((L1-Eta(ph(tt)) - inter_p(tt))/sigma_u(tt),(U1-Eta(ph(tt))-inter_p(tt))/sigma_u(tt));
    end
    
    % update the elements of edge probabilities
    u = unifrnd(0,nu(ph));
    
    % update uijs and stick-breaking random variables
    for h=1:kjs+10
        nus(h) = betarnd(1 + sum(ph==h), aa + sum(ph>h))';
        nu(h) = nus(h)*prod([1 1-nus(1:h-1)]);
        if sum(nu(1:h)) > 1-min(u), break, end
    end
    
    % update allocation to atoms under DP
    pih = -100*ones(p*(p-1)/2,h);       % probabilities of allocation to each atom for each subject
    R = zeros(p*(p-1)/2,h);             % indexes which atoms are available to each subject (depends on their u)
    for l = 1:h
        ind1 = u<nu(l);
        R(ind1,l)=1;
        pih(ind1,l) = -0.5*(1/sigma_u(ind1).^2).*(sum((u_star(ind1,:)- inter_p(ind1,1)*z' - Eta(l) ).^2,2))';
    end
    pih = R.*exp(pih - repmat(max(pih')',[1 h]));
    pih = pih./repmat(pih*ones(h,1),[1 h]);                  % normalize
    pih = [zeros(p*(p-1)/2,1) cumsum(pih')'];
    r = unifrnd(0,1,[p*(p-1)/2,1]);
    for l = 1:h
        ind1 = r>pih(:,l) & r<=pih(:,l+1);
        ph(ind1) = l;
    end
    kjs = max(ph);
    
    res = (sum((u_star - inter_p*z' - Eta(ph)*ones(1,N)).^2,2));
    
    phi = gamrnd( (t_df+1)/2, 1./(0.5*(t_df + res/sigma2_tilde) ) );
    sigma_u = sqrt(sigma2_tilde./phi);
    
    %-- updating the regression coefficients under global-local shrinkage
    %priors
    for h = 1:max(ph)
        if(sum(ph==h)>0)
            V = 1/( 1/(Sigma) + sum(1./sigma_u(ph==h).^2) );
            V = (V+V')/2;
            E =  V*sum( (1./sigma_u(ph==h).^2).*(sum( u_star(ph==h,:)- inter_p(ph==h,1)*z',2 )) );
            atom = normrnd(E, sqrt(V)  );
            Eta(h) = atom;
            fold(h) = 1;
            
        end
    end
    
    %-- updating global shrinkage component
    Sigma = 1/gamrnd(as + max(ph)/2, 1/(bs + sum(Eta(1:max(ph)).^2)) );
    %   Sigma = eye(N);
    
    for tt=1:p*(p-1)/2
        
        ind1 =  prod(normpdf(u_star(tt,z==1), inter_p(tt) + Eta(ph(tt)),sigma_u(tt) ) )*normpdf(inter_p(tt),0,zeta_intp);
        ind0 = prod(normpdf(u_star(tt,z==1), Eta(ph(tt)),sigma_u(tt) ) );
        pind = pi_inc*ind1/((1-pi_inc)*ind0 + pi_inc*ind1);
        if(pind>unifrnd(0,1))
            Vi = 1/( 1/(zeta_intp.^2) + sum(z)*(1/sigma_u(tt))^2);
            Ei = Vi*(1/sigma_u(tt))^2*sum(u_star(tt,z==1)' -  Eta(ph(tt) )' );
            inter_p(tt) = normrnd(Ei,sqrt(Vi));
        else
            inter_p(tt) = 0;
        end
    end
    
    pi_inc = betarnd(api + sum(inter_p>0), bpi + p*(p-1)/2 - sum(inter_p>0) );
    
    %-- updating local shrinkage component
    zeta_intp = sqrt(1./gamrnd(az + 0.5*p*(p-1)/2,1./(bz + 0.5*sum((inter_p.^2)))) );
    
    %--- updating subject specific random effects
    gamma = 10^(-5)*ones(p,nsub,N);
    
    ne = zeros(N,1);
    for nn=1:N
        ne(nn) = sum(sum(triu(reshape(Edge_arr(:,:,nn),[p,p])))) - p;
    end
    
    % Save storage for trace plots
    for nn=1:N
        Sig_save_all(:,(nn*p-p+1):nn*p,iter) = reshape(Sig_arr(:,:,nn),[p,p]) ;
        C_save_all(:,(nn*p-p+1):nn*p,iter) = reshape(C_arr(:,:,nn),[p,p]);
        tau_save_all(:,(nn*p-p+1):nn*p,iter) = reshape(tau_arr(:,:,nn),[p,p]);
    end
        
        % If done with burnin, store the iteration results
    if iter > burnin
        Etaout(:,iter-burnin) = Eta(ph);
        Intout(:,iter-burnin) = inter_p;
        phiout(:,iter-burnin) = phi;
        edgeprob_save(:,:,iter-burnin) = prob;
        for nn=1:N
            Sig_save(:,(nn*p-p+1):nn*p,iter-burnin) = reshape(Sig_arr(:,:,nn),[p,p]) ;
            C_save(:,(nn*p-p+1):nn*p,iter-burnin) = reshape(C_arr(:,:,nn),[p,p]);
            Edge_save(:,(nn*p-p+1):nn*p,iter-burnin) = reshape(Edge_arr(:,:,nn),[p,p]);
            %tau_save(:,(nn*p-p+1):nn*p,iter-burnin) = reshape(tau_arr(:,:,nn),[p,p]);
        end
    end
        
    % If running GUI version, update waitbar and change plot
    if ~isScriptVersion
        
        % Get the group requested by the user
        groupInd = get(findobj('tag', 'groupInd'), 'value');

        %%% Update trace plots
        % Precision Matrix Trace Plot
        sel1 = get(findobj('tag', 'omegaInd1'), 'value');
        sel2 = get(findobj('tag', 'omegaInd2'), 'value') + p*(groupInd-1);
        sel2Element = get(findobj('tag', 'omegaInd2'), 'value');
        axes(findobj('tag','iterChangeOmega'));
        set(gca,'NextPlot','replacechildren');
        plot( squeeze(C_save_all(sel1, sel2, 1:iter)) );
        %xticks([1:iter]);
        if iter > burnin
            maxval = max(squeeze(C_save_all(sel1, sel2, 1:iter)));
            minval = min(squeeze(C_save_all(sel1, sel2, 1:iter)));
            ylim([minval,maxval]);
            line( [burnin, burnin], [minval, maxval], 'color', 'red' ) ;
        end
        title( ['Trace plot of precision matrix element ' num2str(sel1) ',' num2str(sel2Element)...
            ', Group ' num2str(groupInd)]);
        drawnow;
        % Covariance Matrix Trace Plot
        sel1 = get(findobj('tag', 'sigmaInd1'), 'value');
        sel2 = get(findobj('tag', 'sigmaInd2'), 'value') + p*(groupInd-1);
        sel2Element = get(findobj('tag', 'sigmaInd2'), 'value');
        axes(findobj('tag','iterChangeSigma'));
        set(gca,'NextPlot','replacechildren');
        plot( squeeze(Sig_save_all(sel1, sel2, 1:iter)) );
        %xticks([1:iter]);
        if iter > burnin
            maxval = max(squeeze(Sig_save_all(sel1, sel2, 1:iter)));
            minval = min(squeeze(Sig_save_all(sel1, sel2, 1:iter)));
            ylim([minval,maxval]);
            line( [burnin, burnin], [minval, maxval], 'color', 'red' ) ;
        end
        title( ['Trace plot of covariance matrix element ' num2str(sel1) ',' num2str(sel2Element) ...
            ', Group ' num2str(groupInd)]);
        drawnow;
        % Tau Trace Plot
        sel1 = get(findobj('tag', 'tauInd1'), 'value');
        sel2 = get(findobj('tag', 'tauInd2'), 'value') + p*(groupInd-1);
        sel2Element = get(findobj('tag', 'tauInd2'), 'value');
        axes(findobj('tag','iterChangeTau'));
        set(gca,'NextPlot','replacechildren');
        plot( squeeze(tau_save_all(sel1, sel2, 1:iter)) );
        %xticks([1:iter]);
        if iter > burnin
            maxval = max(squeeze(tau_save_all(sel1, sel2, 1:iter)));
            minval = min(squeeze(tau_save_all(sel1, sel2, 1:iter)));
            if (minval == maxval)
                maxval = 0.1;
            end
            ylim([minval,maxval]);
            line( [burnin, burnin], [minval, maxval], 'color', 'red' ) ;
        end
        title( ['Trace plot of tau matrix element ' num2str(sel1) ',' num2str(sel2Element)...
            ', Group ' num2str(groupInd)]);
        drawnow;
    end
    
end % end of iter loop

plot(reshape(C_save(1,2,:),[1,nmc]),'-')

% Post processing steps for graph estimation & variable selection
Omega_est = zeros(p,p,N); Edge_est = zeros(p,p,N); probEdge = zeros(p,p,N);
for nn=1:N
    Est1 = mean(C_save(:,:,1:nmc),3);
    Edge1 = mean(Edge_save,3);
    
    Omega_est(:,:,nn) = reshape(Est1(:,(nn*p-p+1):nn*p),[p,p]);
    Edge_est(:,:,nn) = reshape(Edge1(:,(nn*p-p+1):nn*p),[p,p]);
    tt=0;
    for i=1:p
        if(i<p)
            for j=(i+1):p
                tt = tt+1;
                probEdge(i,j,nn) = mean(reshape(edgeprob_save(nn,tt,1:nmc),[1,nmc]));
                probEdge(j,i,nn) = probEdge(i,j,nn);
            end
        end
    end
end

% should be able to remove following block of code
rho_est11 = 2*eye(p) - Omega_est(:,:,1)./sqrt(diag(Omega_est(:,:,1))*diag(Omega_est(:,:,1))');
Adj_est11 = (abs(rho_est11)>0.1);
Omega_est11 = Omega_est(:,:,1).*Adj_est11;
[ee,ev] = eig(reshape(Omega_est11,[p,p]));
Omega_est11 = Omega_est11 + (0.1 -min(diag(ev)))*eye(p);
rho_est12 = 2*eye(p) - Omega_est(:,:,2)./sqrt(diag(Omega_est(:,:,2))*diag(Omega_est(:,:,2))');
Adj_est12 = (abs(rho_est12)>0.1);
Omega_est12 = Omega_est(:,:,2).*Adj_est12;
[ee,ev] = eig(reshape(Omega_est12,[p,p]));
Omega_est12 = Omega_est12 + (0.1 -min(diag(ev)))*eye(p);

precMat = zeros(size(Omega_est));
pCorrMat = zeros(size(Omega_est));
adjMatrix = zeros(size(Omega_est));
for iGroup = 1:N
    pCorrMat(:,:,iGroup) = 2*eye(p) -...
        Omega_est(:,:,iGroup)./sqrt(diag(Omega_est(:,:,iGroup))*...
        diag(Omega_est(:,:,iGroup))');
    adjMatrix(:,:,iGroup) = (abs( pCorrMat(:,:,iGroup) )>0.01);
    precMatTemp = Omega_est(:,:,iGroup) .* adjMatrix(:,:,iGroup);
    [ee,ev] = eig( reshape(precMatTemp,[p,p]) );
    precMat(:,:,iGroup) = precMatTemp + (0.1 -min(diag(ev)))*eye(p);
end

resultName = [outputFolder '/' prefix '_mcmc_results.mat']

% Save everything if user-requested, otherwise only save the precision
% matrix, partial correlation matrix, and adjacency matrix
if get(findobj('tag', 'saveMCMCBox'), 'value')
    save(resultName, 'precMat', 'pCorrMat', 'adjMatrix', 'C_save', 'Sig_save',...
        'Edge_save');
else
    save(resultName, 'precMat', 'pCorrMat', 'adjMatrix');
end

