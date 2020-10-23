# Neural Analyzer
![Build Status](https://travis-ci.org/norbit8/Neural_Analyzer.svg?branch=master)

 <img src="https://raw.githubusercontent.com/norbit8/Neural_Analyzer/master/essentials/neuralSpikesExample.png" width="800" />

A python package that includes methods for decoding, and plotting neural activity.
Specifically used in the expirement where we sample neural activity from the [Basal ganglia](https://en.wikipedia.org/wiki/Basal_ganglia) and the [cerebellum](https://en.wikipedia.org/wiki/Cerebellum) (Using a *Macaca fascicularis* monkeys) while the monkeys are targeting moving rectangles.

In the following project we build a decoder where given a vector of spikes of neural activity we decode the direction of the eye.

All of the work is based on machine learning algorithms (specifically KNN) which we train using the given data.
First, we suggest you to use `conda` in order to reconstruct our work environemnt. 
### Usage
**1)** First you need to [download](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) anaconda (conda).    
       In order to verify that you do have conda, please type in the terminal ```conda --version```, and look for ```conda x.x.x```.
       
**2)** To recreate the environment you can do the following:

    conda env create -f env.yml
(The env.yml file is located at essentials/env.yml)
       
**3)** Activate the environment like so:

    conda activate neural_analyzer

**4)** Now, we can run python:

    python
       
**5)** Import the modules:

    Import decoder
    Import graphs

**6)** Create an instance of the decoder class and the graphs class:

    dec = decoder.decoder(**kwargs)
    Graph - no need, static lib
       
To see more information please use the ```dec.help()``` method, or ```g.help()``` method.

### Important notes
   
-Matlab files should look likes this "populationName#cellName for example CRB#4391.mat"
   
-When calling the class decoder, Make sure that the directory of the input data have the 
    directories 'PURSUIT'/'SACCADE'/'PURSUIT_FRAGMENTS'/'SACCADE_FRAGMENTS'.
    
-For matching cells one should make sure that only the matching cells are in the input directories for the decoder.

### "out" folder's structure

 <img src="https://raw.githubusercontent.com/norbit8/Neural_Analyzer/master/essentials/Folder_Diagram.png" width="800" />
 
 ### Documentation
 
- [decoder class](https://github.com/norbit8/Neural_Analyzer/blob/master/essentials/decoder_instructions) 
 
- [graphs class](https://github.com/norbit8/Neural_Analyzer/blob/master/essentials/graphs_docs)

