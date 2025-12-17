-- Tabla de leads (Clientes Potenciales)
CREATE TABLE IF NOT EXISTS leads (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) NOT NULL,
  phone VARCHAR(50) NOT NULL,
  age VARCHAR(20),
  interest VARCHAR(100),
  smile_style VARCHAR(50) NOT NULL,
  result_image_url TEXT,
  status VARCHAR(20) DEFAULT 'processing', -- processing, completed, failed
  source VARCHAR(50) DEFAULT 'smile_simulator',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  processed_at TIMESTAMP WITH TIME ZONE,
  contacted_at TIMESTAMP WITH TIME ZONE,
  converted_at TIMESTAMP WITH TIME ZONE,
  notes TEXT
);

-- Índices para búsqueda rápida y dashboard
CREATE INDEX IF NOT EXISTS idx_leads_email ON leads(email);
CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status);
CREATE INDEX IF NOT EXISTS idx_leads_created_at ON leads(created_at DESC);

-- Tabla de conversiones (Opcional para futura analítica)
CREATE TABLE IF NOT EXISTS conversions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  lead_id UUID REFERENCES leads(id),
  treatment_type VARCHAR(100),
  estimated_value DECIMAL(10,2),
  actual_value DECIMAL(10,2),
  converted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Configuración de seguridad (RLS) - Opcional pero recomendado
ALTER TABLE leads ENABLE ROW LEVEL SECURITY;

-- Política: Permitir inserción pública (anon key) pero solo lectura autenticada
CREATE POLICY "Allow public insert" ON leads FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow service role full access" ON leads USING (true) WITH CHECK (true);
