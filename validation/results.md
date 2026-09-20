# Classifier validation

Gold set: 120 tickets labelled by reading the full message and agent note (see `gold_labels.csv` and the docstring of `validate_sample.py` for the rules).

| System | Split | n | Issue exact | Category | Defect acc. | Defect precision | Defect recall |
|---|---|---|---|---|---|---|---|
| Intake tag (baseline) | dev | 80 | - | 85% | 97% | 95% | 95% |
| Rule-based | dev | 80 | 99% | 99% | 100% | 100% (19/19) | 100% (19/19) |
| LLM (gemini-2.5-flash-lite) | dev | 80 | 95% | 98% | 100% | 100% (19/19) | 100% (19/19) |
| Intake tag (baseline) | holdout | 40 | - | 80% | 100% | 100% | 100% |
| Rule-based | holdout | 40 | 85% | 85% | 95% | 86% (6/7) | 86% (6/7) |
| LLM (gemini-2.5-flash-lite) | holdout | 40 | 100% | 100% | 97% | 88% (7/8) | 100% (7/7) |

LLM calls: 0 live, 120 from cache, 0 failed and fell back to rules.

## Where each system is wrong

### Rule-based: 7 of 120 tickets have an issue or defect error

| Ticket | Split | Tag | Gold issue | Predicted | Gold defect | Pred defect |
|---|---|---|---|---|---|---|
| TK-244281 | dev | Other | product or compatibility question | cannot pair or connect | N | N |
| TK-241311 | holdout | Connectivity | connection keeps dropping | no sound / one side dead | U | Y |
| TK-251511 | holdout | Billing & Payments | refund not received | other | N | N |
| TK-249698 | holdout | Other | order not delivered or delayed | other | N | N |
| TK-240138 | holdout | Warranty & Repair | device or case not charging | other | Y | N |
| TK-249515 | holdout | Delivery & Shipping | wrong item delivered | repair or warranty claim follow-up | N | Y |
| TK-249851 | holdout | App & Firmware | firmware update stuck or failed | other | N | N |

### LLM (gemini-2.5-flash-lite): 5 of 120 tickets have an issue or defect error

| Ticket | Split | Tag | Gold issue | Predicted | Gold defect | Pred defect |
|---|---|---|---|---|---|---|
| TK-248967 | dev | Charging & Battery | device or case not charging | no sound / one side dead | Y | Y |
| TK-241406 | dev | Delivery & Shipping | return pickup not done | order not delivered or delayed | N | N |
| TK-253214 | dev | Connectivity | cannot pair or connect | connection keeps dropping | N | N |
| TK-243550 | dev | Connectivity | cannot pair or connect | connection keeps dropping | N | N |
| TK-249851 | holdout | App & Firmware | firmware update stuck or failed | firmware update stuck or failed | N | Y |

