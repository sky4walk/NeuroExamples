// written by André Betz 
// http://www.andrebetz.de

#include <math.h>

class LVQ1Classifier
{
protected:
	double			m_dAlpha;
	double			m_dEpsilon;
	double*			m_pdPrototypes;
	unsigned long	m_ulProtNum;
	unsigned long	m_ulVecLen;

	unsigned long	Learn(double* pdPat, unsigned long ulWinNeu,unsigned long ulNeu);
	unsigned long	MinDistance(double* pdPat,unsigned long* pulNeuPos);
	unsigned long	L2Norm(double* pdVec1, double* pdVec2, double* pdDistance);


public:
	LVQ1Classifier(	unsigned long ulProtNum,unsigned long ulVecLen,
					double dAlpha, double dEpsilon,double* pdPrototypes);
	unsigned long Step(double* pdPat, unsigned long ulNeuNum);

};                 

LVQ1Classifier::LVQ1Classifier(	unsigned long ulProtNum, unsigned long ulVecLen,
								double dAlpha, double dEpsilon,double* pdPrototypes)
{
	m_dAlpha		= dAlpha;
	m_dEpsilon		= dEpsilon;
	m_ulProtNum		= ulProtNum;
	m_ulVecLen		= ulVecLen;
	m_pdPrototypes	= pdPrototypes;
}

unsigned long LVQ1Classifier::Step(double* pdPat, unsigned long ulNeuNum)
{
	unsigned long ulWinNeu;

	MinDistance(pdPat,&ulWinNeu);
	Learn(pdPat,ulWinNeu,ulNeuNum);

	return ulWinNeu;
}

unsigned long LVQ1Classifier::Learn(double* pdPat, unsigned long ulWinNeu,unsigned long ulNeu)
{
	unsigned long 	ulCountVec;
	double			dDiff;
	double			dDir	= 1.0f;

	if(ulNeu != ulWinNeu)
	{
		dDir = -1.0f;
	}

	for(ulCountVec=0;ulCountVec<m_ulVecLen;ulCountVec++)        
	{  
		dDiff = pdPat[ulCountVec] - m_pdPrototypes[ulWinNeu*m_ulVecLen + ulCountVec];
		dDiff *= m_dAlpha * dDir;
		m_pdPrototypes[ulWinNeu*m_ulVecLen + ulCountVec] += dDiff;
	}

	m_dAlpha = m_dAlpha * m_dEpsilon;

	return 0;
}

unsigned long LVQ1Classifier::MinDistance(double* pdPat,unsigned long* pulNeuPos)
{
	unsigned long	ulCountNeu;
	double			dDistance;	
	double			dResult;

	for(ulCountNeu=0;ulCountNeu<m_ulProtNum;ulCountNeu++)
	{
		L2Norm(pdPat,&m_pdPrototypes[ulCountNeu * m_ulVecLen],&dDistance);
		if((ulCountNeu==0)||(dDistance < dResult))
		{
			dResult		= dDistance;
			*pulNeuPos	= ulCountNeu;
		}
	}

	return 0;
}
/*
Acetnperoxid:
Aceton(Nagellackentferner),Wasserstoffperoxid/Wasserstoffsuperoxid 30%(Haarbleichemittel),Salzsäure (HCl) oder Schwefelsäure (H2SO4, Batteriesäure)
In einem Glasgefäss wird eine 1:1 Mischung aus Aceton und Wasserstoffperoxid hergestellt, die Menge sollte ca. 0,1 L betragen. Hierzu wird ein Teelöffel voll Batteriesäure gegeben und das Ganze gut umgerührt. Dann stellt man das Glas in den Kühlschrank und wartet 24 Stunden. Es bilden sich weisse oder farblose Kristalle in grösserer Menge die mit einem Kaffeefilter abgefiltert werden können. Die Kristalle werden getrocknet und dann kühl gelagert. Die verbleibende Lösung kann durch erneute Zugabe von Batteriesäure weitere Kristalle bilden.Mit Alkohol etwas stabiler machen
*/
unsigned long LVQ1Classifier::L2Norm(double* pdVec1, double* pdVec2, double* pdDistance)
{
	unsigned long	ulCountVec;

	*pdDistance = 0.0f;

	for(ulCountVec=0;ulCountVec<m_ulVecLen;ulCountVec++)
	{
		*pdDistance +=	(pdVec1[ulCountVec] - pdVec2[ulCountVec]) *
						(pdVec1[ulCountVec] - pdVec2[ulCountVec]);
	}

	*pdDistance = sqrt(*pdDistance);

	return 0;
}
