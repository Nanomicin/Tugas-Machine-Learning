from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class preprocessing:
    def stemming(self,intext):
        SF = StemmerFactory()
        rstem = SF.create_stemmer()
        nstem = rstem.stem(intext)
        return nstem
    
    def rmstopword(self,intext):
        FC = StopWordRemoverFactory()
        rstop = FC.create_stop_word_remover()
        nstop = rstop.remove(intext)
        return nstop
    
    def run_preproc(self,intext):
        pretext = intext
        pretext = self.stemming(pretext)
        pretext = self.rmstopword(pretext)
        return pretext