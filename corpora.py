import re
def remove_and_quotes(xml_file, name):
    f=open(xml_file)
    f_new=open(name, "w")
    content=f.readlines()
    #regex=re.compile(r"\d+", re.IGNORECASE)
    for line in content:
        if "&" in line:
            line=re.sub(r"([a-zA-Z])*&(?!amp)([a-zA-Z])*", "&amp;" , line)
            #line=line.replace("&", " &amp; ")
        if "id=" in line:
            #print line
            line=re.sub(r'(\d+)', r'"\1"', line)
            #print line
        #if u'\u2013' in line:
         #   line = re.sub(u'\u2013', '-' , line) 
        f_new.write(line)
    f_new.close()
    f_new=open(name, "r")
    return f_new

"""
Aenderungen: 
1. Xml encoding im orgnal
2. allen ands verbessert
3. id in anfuehrungszeichen gesetzt
4. root element hinzugefuegt im orginal
"""
