python scripts/eval/downstream_task/eval_phase_model.py --datapath '../../../results/CycleGAN_CATARACTS_Cataract101_192pix/2022_10_13-06_59_50/' --modelpath 'phase_model.pth' --id CycleGAN --dev cuda:0
python scripts/eval/downstream_task/eval_phase_model.py --datapath '../../../results/RecycleGAN_CATARACTS_Cataract101_192pix/2022_10_13-06_59_45/' --modelpath 'phase_model.pth' --id RecycleGAN --dev cuda:0
python scripts/eval/downstream_task/eval_phase_model.py --datapath '../../../results/UNIT_CATARACTS_Cataract101_192pix/2022_10_13-08_46_20/' --modelpath 'phase_model.pth' --id UNIT --dev cuda:0
python scripts/eval/downstream_task/eval_phase_model.py --datapath '../../../results/OF_UNIT_CATARACTS_Cataract101_192pix/2022_10_13-08_46_17/' --modelpath 'phase_model.pth' --id OF-UNIT --dev cuda:0
python scripts/eval/downstream_task/eval_phase_model.py --datapath '../../../results/MotionUNIT_CATARACTS_Cataract101_192pix/2022_10_13-08_46_18/' --modelpath 'phase_model.pth' --id MT-UNIT --dev cuda:0