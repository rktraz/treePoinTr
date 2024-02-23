"""
script for performing helios++ simulations, using pyhelios
multiple scan positions around one tree model
change "tree" scenepart: loop through .obj files in folder, do the same simulation for each tree


make sure that the HELIOS++ root directory is either the current working folder or added to the PATH environment variable, e.g. using set PATH=%PATH%;C:\path\to\helios-plusplus-win
"""
import sys
import os
import glob
import numpy as np
import time
import shutil
from pathlib import Path, PurePath
# PyHelios import
import pyhelios
from pyhelios import outputToNumpy, SimulationBuilder
from pyhelios.util import scene_writer


# Set logging.
#pyhelios.loggingQuiet()
pyhelios.loggingVerbose()

# Set seed for default random number generator.
pyhelios.setDefaultRandomnessGeneratorSeed("123")

# print current helios version
print(pyhelios.getVersion())

path = "./data/sceneparts/grovetrees/"

for file in glob.glob(path +"*.obj"):

    tree_filename = os.path.basename(file)


    tree_part = scene_writer.create_scenepart_obj(Path(file), up_axis="z")
    scene_id = tree_filename # replace with filename in loop
    scene_content = scene_writer.build_scene(scene_id=scene_id, name="center tree", sceneparts=tree_part)

    scene_folder = "./data/scenes/grovetrees/"
    scene_file = scene_folder+"scene_"+tree_filename+".xml"  # insert tree filename in loop


    with open(scene_file, "w") as f:
        f.write(scene_content)



    survey_file = "./data/surveys/tls_complete_sim_centertree_"+tree_filename+".xml"

    legs =  """
            <leg>
                <platformSettings x="0" y="8" onGround="true" />
                <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="0" headRotateStop_deg="360" pulseFreq_hz="300000" trajectoryTimeInterval_s="1.0"/>
            </leg>
            <leg>
                <platformSettings x="8" y="0" onGround="true" />
                <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="0" headRotateStop_deg="360" pulseFreq_hz="300000" trajectoryTimeInterval_s="1.0"/>
            </leg>
            <leg>
                <platformSettings x="-8" y="0" onGround="true" />
                <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="0" headRotateStop_deg="360" pulseFreq_hz="300000" trajectoryTimeInterval_s="1.0"/>
            </leg>
            <leg>
                <platformSettings x="0" y="-8" onGround="true" />
                <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="0" headRotateStop_deg="360" pulseFreq_hz="300000" trajectoryTimeInterval_s="1.0"/>
            </leg>
            <leg>
                <platformSettings x="12" y="12" onGround="true" />
                <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="1" headRotateStop_deg="360"  pulseFreq_hz="300000" trajectoryTimeInterval_s="1.0"/>
            </leg>
            <leg>
                <platformSettings x="12" y="-12" onGround="true" />
                <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="1" headRotateStop_deg="360" pulseFreq_hz="300000" trajectoryTimeInterval_s="1.0"/>
            </leg>		
            <leg>
                <platformSettings x="-12" y="12" onGround="true" />
                <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="1" headRotateStop_deg="360" pulseFreq_hz="300000" trajectoryTimeInterval_s="1.0"/>
            </leg>	
            <leg>
                <platformSettings x="-12" y="-12" onGround="true" />
                <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="1" headRotateStop_deg="360" pulseFreq_hz="300000" trajectoryTimeInterval_s="1.0"/>
            </leg>	
            <leg>
                <platformSettings x="6" y="3" z="7" />
                <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="1" headRotateStop_deg="360" pulseFreq_hz="100000" trajectoryTimeInterval_s="1.0"/>
            </leg> 
            <leg>
                <platformSettings x="3" y="-6" z="7" />
                <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="1" headRotateStop_deg="360" pulseFreq_hz="100000" trajectoryTimeInterval_s="1.0"/>
            </leg>             
            <leg>
                <platformSettings x="-6" y="-3" z="7" />
                <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="1" headRotateStop_deg="360" pulseFreq_hz="100000" trajectoryTimeInterval_s="1.0"/>
            </leg>  
            <leg>
                <platformSettings x="-3" y="6" z="7" />
                <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="1" headRotateStop_deg="360" pulseFreq_hz="100000" trajectoryTimeInterval_s="1.0"/>
            </leg>             
    """

    survey_content = """<?xml version="1.0" encoding="UTF-8"?>
    <document>
        <!-- Default scanner settings: -->
        <scannerSettings id="profile1" active="true" pulseFreq_hz="100000" scanFreq_hz="120" scanAngle_deg="100" headRotatePerSec_deg="10.0"/>
        <survey name="tls_sims_centertree" platform="data/platforms.xml#tripod" scanner="data/scanners_tls.xml#riegl_vz400" scene="{}#{}">
            <FWFSettings binSize_ns="0.2" beamSampleQuality="3" />
        {}
        </survey>
    </document>

    """.format(scene_file, scene_id, legs)


    with open(survey_file, "w") as f:
        f.write(survey_content)
        
        

    # Build simulation parameters
    simBuilder = pyhelios.SimulationBuilder(str(survey_file), 'assets/','output/')
    simBuilder.setNumThreads(0)
    simBuilder.setLasOutput(False)
    simBuilder.setZipOutput(False)
    simBuilder.setCallbackFrequency(0)  # Run without callback
    simBuilder.setFinalOutput(False)     # Return output at join
    simBuilder.setExportToFile(True)   # Disable export pointcloud to file
    simBuilder.setRebuildScene(False)

    outdir = "./output/py_simulations/"+tree_filename
    if not os.path.exists(outdir):
        os.makedirs(outdir)
               
    simBuilder.setOutputDir(outdir)



    sim = simBuilder.build()

   
    # Start the simulation.
    start_time = time.time()
    sim.start()
    
    if sim.isStarted():
        print('Simulation has started!')
        
     

    while sim.isRunning():
        duration = time.time()-start_time
        mins = duration // 60
        secs = duration % 60
        print("\r"+"Simulation is running since {} min and {} sec. Please wait.".format(int(mins), int(secs)), end="")
        time.sleep(10)   
        
    if sim.isFinished():
        print('Simulation has finished.') 
        
print("all files done")        