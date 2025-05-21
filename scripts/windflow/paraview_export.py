from paraview.simple import *
import argparse

parser = argparse.ArgumentParser(description='Process OpenFOAM data with ParaView')
parser.add_argument('--foam_proj', required=True, help='Path to OpenFOAM case file (.foam)')
parser.add_argument('--output_csv', required=True, help='Path to output CSV file')
args = parser.parse_args()

simulation_proj = args.foam_proj
output_csv = args.output_csv

openfoam = OpenFOAMReader(registrationName='open.foam', FileName=simulation_proj)
openfoam.MeshRegions = ['internalMesh']
openfoam.CellArrays = ['U', 'nuTilda', 'nut', 'p']

animationScene1 = GetAnimationScene()
animationScene1.UpdateAnimationUsingDataTimeSteps()
UpdatePipeline(time=50.0, proxy=openfoam)

SaveData(output_csv, proxy=openfoam, 
         PointDataArrays=['U', 'nuTilda', 'nut', 'p'],
         CellDataArrays=['U', 'nuTilda', 'nut', 'p'],
         FieldDataArrays=['CasePath'])
