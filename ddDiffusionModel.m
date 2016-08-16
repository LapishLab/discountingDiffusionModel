function [chSel,respLat,trkiBound,trkAccumVal]=ddDiffusionModel(steps, diffusion, nRep, delayTime, plotData, kValue, scaleDriftBias,scaleOriginBias)
%% Output variables 
% chSel                             % Boundary found on each trial
% respLat                           % Latency to find boundary 
% trkiBound                         % Gives the value of the immediate/trial
% trkAccumVal                       % Tracks the number of consecutive boundaries found

%% Input variables (values from Linsenbardt et al., 2016)
% clear all
% nRep=500;                           % Number of r(trials) 
% steps=5000;                         % Number of steps in the diffusion model
% diffusion=0.04;                     % Minimizes omits for 5000 steps
% delayTime=0;                        % Time of the delayed 'reward'(boundary)
% plotData=1;                         % Would you like to plot the data
% kValue=0.3009;                      % k-value to construct boundaries 
% scaleOriginBias=0;                  % Origin bias (see Linsenbardt et al., 2016)
% scaleDriftBias=0;                   % Scale bias (see Linsenbardt et al., 2016)

%% Setting boundaries by v=A/1+(kD), where boundary = 1/v
% setting immediate boundary vector
A=[1:6];                           % vary from 1 to 6 pellets
immediateTime=0;                   % immediate delay is zero
v=A./(1+kValue.*(immediateTime));  % obtains relative value of the reward(v)
iBound=1./v;                       % vector of immediate reward boundary distances (more valuable = closer to origin)
iValCt=3;                          % counter to adjust immediate reward boundary distances (starts at midpoint, 3)

% setting delay boundary,
A=6;                               % delay is always worth 6 pellets
v=A./(1+kValue*(delayTime));       % temporal delay to larger reward
dBound=-1./v;                      % scalar for delayed reward boundary distance

%% Creating bias in initial portion of walk to model "memory trace" decays quasi-exponentially
x=[0:steps];
walkBias=(scaleDriftBias*dBound).*(1-0.05).^x;

%% Create function to migrate origin based on number of consecutive choices
% % origin drift is sigmoidal and based on number of consecutive choices
xVals=[-100:5:100];
y=(xVals./sqrt(500+xVals.^2))+1;
originBias=scaleOriginBias.*(y./max(y));
originBias(1)=0;  % there should be no update on first seclection
originBiasCt=1;

%% setting accumlators to 1
delAccum=1;
immAccum=1;

%% Loop on number of trials (nRep) 
for XX=1:nRep;
    %% detect what the last choice was, and update accumulators: 
    % % this provides index to select weight vector from
    if XX ~=1; % Skip first trial, use default
        if chSel(XX-1)==-1;
            delAccum=delAccum+1;
            immAccum=immAccum-1;
        elseif chSel(XX-1)==1
            immAccum=immAccum+1;
            delAccum=delAccum-1;
        else
            immAccum=immAccum-1;
            delAccum=delAccum-1;
        end;
    end;
    
    % make sure that accumlators don't leave range of weightVectors
    if delAccum < 1; delAccum=1; end;
    if delAccum > length(originBias); delAccum=length(originBias); end;
    if immAccum < 1; immAccum=1; end;
    if immAccum > length(originBias); immAccum=length(originBias); end;
    accumVal=immAccum-delAccum;
    if accumVal==0;accumVal=1;end; 
    
    % track values across trials
    trkAccumVal(XX)=accumVal;
    trkImmAccum(XX)=immAccum;
    trkDelAccum(XX)=delAccum;   
    
    %%  Random Walk
    stepsM=steps+1;
    A=nan(1,stepsM);
    if XX~=1;
        if accumVal>0 % found immediate
            A(:,1)=originBias(accumVal)*iBound(iValCt); % biases orgin towards iBound (immediate)
        end;
        if accumVal<0 % found delayed
            A(:,1)=originBias(abs(accumVal))*dBound; % biases orgin towards dBound (delay)
        end;
    end;
    if XX==1; A(:,1)=0; end; % start at origin
    % perform walk
    for j=2:stepsM;
        A(1,j)=A(1,j-1)+diffusion*pearsrnd(walkBias(j-1),1,0,3,1,1);
    end;
    
    trkiBound(XX)=iBound(iValCt);
    trkA(XX,:)=A;

    %% Did walk find target?
    walks(XX,:)=A;
    
    iIdx=find(A>=iBound(iValCt),1);
    dIdx=find(A<=dBound,1);
    
    if isempty(iIdx);iIdx=steps+1;end;
    if isempty(dIdx);dIdx=steps+1;end;
        
    chSel(XX)=0; % assumes omission and sets to zero

    if iIdx<dIdx; % Immediate found first
        chSel(XX)=1;    
        if iBound(iValCt)~=max(iBound); % increase to max iBound but not higher
            iValCt=iValCt-1;            % Increase location of immediate boundary 
        end;
    end;
    
    if dIdx<iIdx; % Delayed found first.         
        chSel(XX)=-1;
        if iBound(iValCt)~=min(iBound);
           iValCt=iValCt+1; % Decrease location of immediate boundary
        end; 
    end;
    
    if dIdx==iIdx;
        if iBound(iValCt)~=min(iBound);
           iValCt=iValCt+1; % Decrease location of immediate boundary
        end;  
    end;
    
    
    %%  Extract Responce Latencies: Number of steps taken to find the target
    if chSel(XX)==-1;
        respLat(XX)=dIdx;
    end;
    if chSel(XX)==1;
        respLat(XX)=iIdx;
    end;
    if chSel(XX)==0;
        respLat(XX)=nan;
    end;
end;

%% Plotting tools
if plotData==1
    subplot(2,2,1);
    plot(chSel,'ko-');
    title('Choice (-1=Delay,0=Omission, 1=immediate)');
    ylabel('Choice');
    xlabel('Trial Number');
    ylim([-1.1 1.1]);
    subplot(2,2,2);
    k=find(chSel==1);
    plot(k,trkiBound(k),'bo');hold on;
    k=find(chSel==-1);
    plot(k,trkiBound(k),'ro');hold on;
    title('Immediate Boundary Value (blue= immediate, red=delay)');
    ylabel('iValue (Distance from origin)');
    xlabel('Trial Number');
    subplot(2,2,3);
    k=find(chSel==1);
    plot(k,respLat(k),'bo');hold on;
    k=find(chSel==-1);
    plot(k,respLat(k),'ro');hold on;
    title('Response Latency (blue= immediate, red=delay)');
    ylabel('RespLatency (bins)');
    xlabel('Trial Number');  
    subplot(2,2,4);
    hist(chSel);
    title('Number of Immediate, Omitted, Delay choices');
    set(gca,'XTick',[-1 0 1],'XTickLabel',{'Delay','Omit','Immediate'});
    xlim([-1.5 1.5]);
    ylabel('Counts');
end;

%%    (c) 2016 Christopher C. Lapish PhD Indiana University 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
