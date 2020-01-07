# NLP-Projects

## Project 1

  * Change the *dir* variable according to the test environment
  * Please run ***naive-bow.jl*** file
  * Play with constants as you would like to 

## Project 2

  * There are two main files which are totally ready to be tested
  * The only difference for both files is the array type which can be manipulated with respect to the test environment
  
  	* CPU version of the code is worked on a CPU machine, but would be a suffering choice to be tested
	
		> *GPU version is really fast, but has memory allocation issues which we searched a lot, but could not offer a perfectly working solution* **(SOLVED!)**
	
	* *Restricted the memory used by **KnetArray** which causes a memory shortage when the randomness needed*
		* Use the latest version of the limited test file for the final training test
		
			> Commenting in the previous test cases would be sufficient to test all project
## Project 3

* There are two Julia files which are ready to be tested in CPU and GPU environments, specifically.
	* Test result in CPU version is more accurate since we have changed the *batch size* in order to fit GPU arrays in to memory.
	* In order to provide a fluent run, test cases which halts the program as they are too specific, are commented out. Instead, results are printed out for each test cases.
* For convenience, the jupyter file, that was run on *Google Colab* environment that consists of a K80 GPU instance, provided.
* You can access the trained model via following Drive link:
	> https://drive.google.com/open?id=1JFIbSjknzDBLDI-uSeu5XueFAj_-AxFe
	
## Project 4

~* There are two Julia files which are ready to be tested in CPU and GPU environments, specifically.
	* GPU file is cleared from tests, which focuses on traning.
	* Tests can be performed through the main file. (*attn-template*)~
	
* We have provided a jupyter notebook which trained in a GPU machine and fulfills all tests. 
	* You can access pretrained model via the same drive link above.
