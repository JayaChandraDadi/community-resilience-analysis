# run_all.py
from paper1_rcn import paper1_rcn_analysis
from paper2_triangle import paper2_resilence_triangle

def main():
    print("Running Paper 1 RCN analysis...")
    paper1_rcn_analysis.run()

    print("Running Paper 2 resilience triangle analysis...")
    paper2_resilence_triangle.run()

if __name__ == "__main__":
    main()
