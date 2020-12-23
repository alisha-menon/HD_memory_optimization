
load('sensordata1.mat')
low = min(inputs);
high = max(inputs);
s = (inputs-low)/(high-low);

Fs=31;
%%Fi%%eroutthehighfrequencynose%%
%%%%%%%%%%XXX%%%%%%%%%%%%%%%%%X%%%%%%%%
lgsr=length(s);
lgsr2=lgsr/2;
t=(1:lgsr)/Fs;
[b,a]=ellip(4,0.1,40,4*2/Fs);
[H,w]=freqz(b,a,lgsr);
sf=filter(b,a,s);
sf_prime=diff(sf);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%There is ringing in the signal so the first 35 points are excluded
sf_prime35=sf_prime(35:length(sf_prime));
%Set a threshhold to define significant startle
%thresh=0.005;
thresh=0.003;
vector=sf_prime35;
overthresh=find(vector>thresh);
overthresh35=overthresh+35;
%the true values of the segment
gaps=diff(overthresh35);
big_gaps=find(gaps>31);

iend=[];
ibegin=[];
for i=1:length(big_gaps)
iend=[iend overthresh35(big_gaps(i))];
ibegin=[ibegin overthresh35(big_gaps(i)+1)];
end

overzero=find(sf_prime>0);
zerogaps=diff(overzero);
z_gaps=find(zerogaps>1);
iup=[];
idown=[];
for i=1:length(z_gaps)
idown=[idown overzero(z_gaps(i))];
iup=[iup overzero(z_gaps(i)+1)];
end
% find up crossing closest to ibegin
new_begin=[];
for i=1:length(ibegin)
temp=find(iup<ibegin(i));
choice=temp(length(temp));
new_begin(i)=iup(choice);
end

% to find the end of the startle, find the maximum between startle
% beginnings
new_end=[];

for i=1:(length(new_begin)-1)
startit=new_begin(i)
endit=new_begin(i+1)
[val, loc]=max(s(startit:endit))
new_end(i)=startit+loc
end

if (length(new_begin)>0)
lastbegin=new_begin(length(new_begin));
[lastval,lastloc]=max(s(lastbegin:length(s)-1));
new_end(length(new_begin))=new_begin(length(new_begin))+lastloc;
end

smag=[]; %initialize a vector of startle magnitudes
sdur=[]; %initialize a vector of startle durations
for i=1:length(new_end)
sdur(i)=new_end(i)-new_begin(i);
smag(i)=s(new_end(i))-s(new_begin(i));
end

s_freq=length(ibegin);