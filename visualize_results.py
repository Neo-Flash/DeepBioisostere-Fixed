#!/usr/bin/env python3
"""
Visualization script for DeepBioisostere molecule generation results
Displays all generated molecules with their properties
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import math
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Set English font and style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def mol_to_image(smiles, molSize=(200, 200)):
    """Convert SMILES to molecular structure image"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=molSize)
        return img
    except:
        return None

def plot_all_molecules():
    """Plot all molecules with their properties"""
    
    try:
        # Read only one CSV file
        csv_file = "generation_result.csv"
        df = pd.read_csv(csv_file)
        
        # Create output folder based on CSV filename
        folder_name = os.path.splitext(csv_file)[0]  # Remove .csv extension
        output_folder = os.path.join(os.getcwd(), folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print("=== Molecule Generation Results ===")
        print(f"Generated molecules: {len(df)} molecules")
        print(f"Output folder: {output_folder}")
        
        # Sort by predicted probability
        df_sorted = df.sort_values('PREDICTED-PROB', ascending=False).reset_index(drop=True)
        
        # 1. Plot all molecules in one figure
        print("\n=== All Generated Molecules ===")
        n_molecules = len(df_sorted)
        cols = 6  # 6 columns for more compact display
        rows = math.ceil(n_molecules / cols)
        
        # Create figure for all molecules
        fig = plt.figure(figsize=(24, 4*rows))
        # fig.suptitle('All Generated Molecules (Sorted by Predicted Probability)', fontsize=16, fontweight='bold')
        
        for i, (idx, mol_data) in enumerate(df_sorted.iterrows()):
            ax = plt.subplot(rows, cols, i+1)
            
            # Get molecule image
            img = mol_to_image(mol_data['GEN-MOL-SMI'])
            if img:
                plt.imshow(img)
                plt.axis('off')
                
                # Create title with all information
                title = (f"#{i+1} Prob:{mol_data['PREDICTED-PROB']:.4f}\n"
                        f"LogP:{mol_data['LOGP']:.2f} MW:{mol_data['MW']:.1f}\n"
                        f"QED:{mol_data['QED']:.2f} SA:{mol_data['SA']:.2f}")
                
                plt.title(title, fontsize=8, pad=5)
            else:
                plt.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center')
                plt.axis('off')
        
        plt.tight_layout()
        molecules_png = os.path.join(output_folder, "all_generated_molecules.png")
        plt.savefig(molecules_png, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {molecules_png}")
        plt.show()
        
        # 2. Property distributions
        print("\n=== Property Distributions ===")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Property Distributions of Generated Molecules', fontsize=14, fontweight='bold')
        
        properties = ['LOGP', 'MW', 'QED', 'SA']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']
        
        for i, (prop, color) in enumerate(zip(properties, colors)):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Plot histogram
            ax.hist(df[prop], bins=20, alpha=0.7, color=color, edgecolor='black')
            ax.set_title(f'{prop} Distribution')
            ax.set_xlabel(prop)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = df[prop].mean()
            std_val = df[prop].std()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax.text(0.02, 0.95, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMin: {df[prop].min():.2f}\nMax: {df[prop].max():.2f}', 
                   transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8), 
                   fontsize=10)
        
        plt.tight_layout()
        distributions_png = os.path.join(output_folder, "property_distributions.png")
        plt.savefig(distributions_png, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {distributions_png}")
        plt.show()
        
        # 3. Top 10 molecules with transformation details
        print("\n=== Top 10 Molecular Transformations ===")
        top_10 = df_sorted.head(10)
        
        # Create a large figure for all transformations
        fig = plt.figure(figsize=(20, 25))
        fig.suptitle('Top 10 Molecular Transformations (by Predicted Probability)', fontsize=16, fontweight='bold')
        
        for i, (idx, mol_data) in enumerate(top_10.iterrows()):
            base_subplot = i * 4 + 1
            
            # Original molecule
            ax = plt.subplot(10, 4, base_subplot)
            img = mol_to_image(mol_data['INPUT-MOL-SMI'])
            if img:
                plt.imshow(img)
            plt.axis('off')
            plt.title(f"Rank {i+1}\nOriginal Molecule", fontsize=10, fontweight='bold')
            
            # Leaving fragment
            ax = plt.subplot(10, 4, base_subplot + 1)
            img = mol_to_image(mol_data['LEAVING-FRAG-SMI'])
            if img:
                plt.imshow(img)
            plt.axis('off')
            plt.title("Leaving Fragment", fontsize=10)
            
            # Inserting fragment
            ax = plt.subplot(10, 4, base_subplot + 2)
            img = mol_to_image(mol_data['INSERTING-FRAG-SMI'])
            if img:
                plt.imshow(img)
            plt.axis('off')
            plt.title("Inserting Fragment", fontsize=10)
            
            # Generated molecule
            ax = plt.subplot(10, 4, base_subplot + 3)
            img = mol_to_image(mol_data['GEN-MOL-SMI'])
            if img:
                plt.imshow(img)
            plt.axis('off')
            title = (f"Generated Molecule\nProb: {mol_data['PREDICTED-PROB']:.4f}\n"
                    f"LogP: {mol_data['LOGP']:.2f}, MW: {mol_data['MW']:.1f}\n"
                    f"QED: {mol_data['QED']:.2f}, SA: {mol_data['SA']:.2f}")
            plt.title(title, fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        transformations_png = os.path.join(output_folder, "top10_transformations.png")
        plt.savefig(transformations_png, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {transformations_png}")
        plt.show()
        
        # Print detailed information table
        print(f"\nDetailed Information:")
        print("-" * 140)
        print(f"{'Rank':<4} {'INPUT-SMI':<35} {'GEN-SMI':<35} {'LEAVING-FRAG':<20} {'INSERT-FRAG':<20} {'PROB':<8} {'LogP':<6} {'MW':<6} {'QED':<6} {'SA':<6}")
        print("-" * 140)
        
        for i, (idx, mol_data) in enumerate(df_sorted.iterrows()):
            input_smi = mol_data['INPUT-MOL-SMI'][:32] + "..." if len(mol_data['INPUT-MOL-SMI']) > 35 else mol_data['INPUT-MOL-SMI']
            gen_smi = mol_data['GEN-MOL-SMI'][:32] + "..." if len(mol_data['GEN-MOL-SMI']) > 35 else mol_data['GEN-MOL-SMI']
            leaving_frag = mol_data['LEAVING-FRAG-SMI'][:17] + "..." if len(mol_data['LEAVING-FRAG-SMI']) > 20 else mol_data['LEAVING-FRAG-SMI']
            insert_frag = mol_data['INSERTING-FRAG-SMI'][:17] + "..." if len(mol_data['INSERTING-FRAG-SMI']) > 20 else mol_data['INSERTING-FRAG-SMI']
            
            print(f"{i+1:<4} {input_smi:<35} {gen_smi:<35} {leaving_frag:<20} {insert_frag:<20} "
                  f"{mol_data['PREDICTED-PROB']:<8.4f} {mol_data['LOGP']:<6.2f} {mol_data['MW']:<6.1f} "
                  f"{mol_data['QED']:<6.3f} {mol_data['SA']:<6.2f}")
        
        # Summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Total molecules generated: {len(df)}")
        print(f"Unique molecules: {df['GEN-MOL-SMI'].nunique()}")
        print(f"Average predicted probability: {df['PREDICTED-PROB'].mean():.6f}")
        print(f"Best predicted probability: {df['PREDICTED-PROB'].max():.6f}")
        
        print("\nProperty Statistics:")
        for prop in properties:
            print(f"{prop}: Mean={df[prop].mean():.3f}, Std={df[prop].std():.3f}, "
                  f"Range=[{df[prop].min():.3f}, {df[prop].max():.3f}]")
            
        # Fragment analysis
        print("\n=== Fragment Analysis ===")
        print("Top 5 Most Common Leaving Fragments:")
        leaving_counts = df['LEAVING-FRAG-SMI'].value_counts().head(5)
        for i, (frag, count) in enumerate(leaving_counts.items(), 1):
            print(f"  {i}. {frag}: {count} times")
            
        print("\nTop 5 Most Common Inserting Fragments:")
        insert_counts = df['INSERTING-FRAG-SMI'].value_counts().head(5)
        for i, (frag, count) in enumerate(insert_counts.items(), 1):
            print(f"  {i}. {frag}: {count} times")
        
        # Export detailed CSV to the same folder
        detailed_csv = os.path.join(output_folder, "detailed_results.csv")
        df_sorted.to_csv(detailed_csv, index=False)
        print(f"\nExported detailed results to: {detailed_csv}")
        
    except FileNotFoundError as e:
        print(f"Result file not found. Please run molecule generation first: {e}")
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("DeepBioisostere Results Visualization")
    print("=" * 50)
    
    # Check if result file exists
    csv_file = "generation_result.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Please run the molecule generation in example.ipynb first.")
        sys.exit(1)
    
    # Run visualization
    plot_all_molecules()
    
    # Get folder name for output summary
    folder_name = os.path.splitext(csv_file)[0]
    
    print("\n" + "=" * 50)
    print("Visualization complete!")
    print(f"All files saved in folder: {folder_name}/")
    print("Generated files:")
    print(f"- {folder_name}/all_generated_molecules.png")
    print(f"- {folder_name}/property_distributions.png") 
    print(f"- {folder_name}/top10_transformations.png")
    print(f"- {folder_name}/detailed_results.csv")