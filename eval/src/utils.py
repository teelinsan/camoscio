
split_token = "### Risposta:"

def generate_prompt(instruction, input=None):
    if input:
        return f"""Di seguito è riportata un'istruzione che descrive un task, insieme ad un input che fornisce un contesto più ampio. Scrivete una risposta che completi adeguatamente la richiesta.

### Istruzione:
{instruction}

### Input:
{input}

### Risposta:"""
    else:
        return f"""Di seguito è riportata un'istruzione che descrive un task. Scrivete una risposta che completi adeguatamente la richiesta.

### Istruzione:
{instruction}

### Risposta:"""