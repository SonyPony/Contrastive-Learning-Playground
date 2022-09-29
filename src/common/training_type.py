from enum import Enum


class TrainingType(Enum):
    LINEAR_EVAL = "lin_eval"
    SUPERVISED_CONTRASTIVE = "sup_con"
    SELF_SUPERVISED_CONTRASTIVE = "ss_con"
    SUPERVISED_CONTRASTIVE_SIG = "sup_con_sig"