function y = acf(x, frequencyBins, signalPowerSpectrum)
% This function calculates the (normalized) autocorrelation function from the sampled spectrum of a signal.
% x: time-domain shift points in which acf will be evaluated
% frequencyBins: vector containing values of all the pilot's frequencies (i.e. 256 for f. band 1 + 256 for f. band 2)
% signalPowerSpectrum: vector of power values allocated to each pilot (i.e. 256 for f. band 1 + 256 for f. band 2)
% NOTE: index 1 of frequencyBins corresponds to index 1 of specSig

f = frequencyBins.';
y = (signalPowerSpectrum) * exp(1i*2*pi*f*x) / sum(signalPowerSpectrum);
return;

