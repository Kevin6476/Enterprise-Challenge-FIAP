# Official business rule for default classification.
# A boleto is defaulted if unpaid (dt_pagamento is null)
# or paid more than DEFAULT_THRESHOLD_DAYS after due date.
DEFAULT_THRESHOLD_DAYS: int = 0
