// src/pages/MachineLearning/MyDatasetPage.jsx
import React, { useState } from 'react';
import dayjs from 'dayjs';
import 'dayjs/locale/pt-br';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Stack,
  Typography,
  Button,
  Grid,
  TextField,
  IconButton,
  Card,
  CardHeader,
  CardContent,
} from '@mui/material';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import AddIcon from '@mui/icons-material/Add';
import SearchIcon from '@mui/icons-material/Search';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import FormatListNumberedIcon from '@mui/icons-material/FormatListNumbered';

export default function MyDatasetPage() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');

  // TODO: substituir por fetch real
  const datasets = [
    {
      id: 1,
      name: 'Protocolo A – Séries Temporais',
      created: '2025-07-10',
      count: 42,
      filters: {
        clients: ['Cliente X', 'Cliente Y'],
        cities: ['São Paulo'],
        protocol: 'Protocolo A',
        analyses: ['Bruto', 'Tratado'],
        calibrations: ['Coliformes totais (UFC/mL)'],
      },
    },
    {
      id: 2,
      name: 'Protocolo A – Features',
      created: '2025-07-12',
      count: 37,
      filters: {
        clients: ['Cliente X'],
        cities: ['Rio de Janeiro'],
        protocol: 'Protocolo A',
        analyses: ['Bruto'],
        calibrations: ['Escherichia coli (NMP/mL)'],
      },
    },
    {
      id: 3,
      name: 'Protocolo B – Séries Temporais',
      created: '2025-07-15',
      count: 19,
      filters: {
        clients: ['Cliente Z'],
        cities: ['Belo Horizonte'],
        protocol: 'Protocolo B',
        analyses: ['Tratado'],
        calibrations: ['Coliformes totais (UFC/mL)', 'Escherichia coli (NMP/mL)'],
      },
    },
    {
      id: 4,
      name: 'Protocolo B – Features',
      created: '2025-07-18',
      count: 58,
      filters: {
        clients: [],
        cities: [],
        protocol: 'Protocolo B',
        analyses: [],
        calibrations: [],
      },
    },
  ];

  const filtered = datasets.filter(ds =>
    ds.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <Box sx={{ p: { xs: 2, sm: 3, md: 4 } }}>
      {/* Header */}
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        spacing={2}
        alignItems="center"
        justifyContent="space-between"
        mb={3}
      >
        <Stack direction="row" spacing={1} alignItems="center">
          <Button variant="outlined" size="small" onClick={() => navigate(-1)}>
            Voltar
          </Button>
          <Typography variant="h4" color="primary">
            Meus Datasets
          </Typography>
        </Stack>
        <Stack direction="row" spacing={1} alignItems="center">
          <TextField
            size="small"
            variant="outlined"
            placeholder="Buscar..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            InputProps={{
              endAdornment: (
                <IconButton size="small">
                  <SearchIcon />
                </IconButton>
              ),
            }}
          />
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => navigate('/machine-learning/create-dataset')}
          >
            Criar Dataset
          </Button>
        </Stack>
      </Stack>

      {/* Grid de cards */}
      <Grid container spacing={3}>
        {filtered.length === 0 ? (
          <Grid item xs={12}>
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="h6" color="text.secondary">
                Nenhum dataset encontrado.
              </Typography>
            </Paper>
          </Grid>
        ) : (
          filtered.map(ds => (
            <Grid item xs={12} sm={6} md={4} key={ds.id}>
              <Card elevation={3} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                {/* Cabeçalho com ícone, título e data */}
                <CardHeader
                  avatar={<InsertDriveFileIcon color="primary" />}
                  title={ds.name}
                  subheader={dayjs(ds.created).format('DD/MM/YYYY')}
                  titleTypographyProps={{ variant: 'subtitle1', noWrap: true }}
                  subheaderTypographyProps={{ variant: 'caption', color: 'text.secondary' }}
                />

                <CardContent sx={{ flexGrow: 1 }}>
                  {/* Quantidade de experimentos */}
                  <Stack direction="row" alignItems="center" spacing={1} mb={2}>
                    <FormatListNumberedIcon fontSize="small" color="action" />
                    <Typography variant="body2">{ds.count} experimentos</Typography>
                  </Stack>

                  {/* Filtros em colunas */}
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">Clientes</Typography>
                      <Typography variant="body2">
                        {ds.filters.clients.length ? ds.filters.clients.join(', ') : '—'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">Cidades</Typography>
                      <Typography variant="body2">
                        {ds.filters.cities.length ? ds.filters.cities.join(', ') : '—'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">Protocolo</Typography>
                      <Typography variant="body2">
                        {ds.filters.protocol || '—'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">Análises</Typography>
                      <Typography variant="body2">
                        {ds.filters.analyses.length ? ds.filters.analyses.join(', ') : '—'}
                      </Typography>
                    </Grid>
                    <Grid item xs={12}>
                      <Typography variant="caption" color="text.secondary">Calibrações</Typography>
                      <Typography variant="body2">
                        {ds.filters.calibrations.length ? ds.filters.calibrations.join(', ') : '—'}
                      </Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          ))
        )}
      </Grid>
    </Box>
  );
}
