function varargout = bjnl(varargin)
% 
% This is the primary function for the BJNL toolbox, which acts to add 
% all needed folders to the path and then opens the GUI window.

    addpath(genpath('examples'))
    addpath(genpath('gui'))
    addpath(genpath('src'))

    main();


