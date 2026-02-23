// src/pages/MachineLearning/MyModelsPage.jsx
import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Stack,
  Typography,
  Button,
  Divider,
} from '@mui/material';

export default function MyModelsPage() {
  const navigate = useNavigate();

  return (
    <Box sx={{ p: { xs: 2, sm: 3, md: 4 } }}>
      <Stack direction="row" spacing={1} alignItems="center" mb={2}>
        <Button variant="outlined" size="small" onClick={() => navigate(-1)}>
          Voltar
        </Button>
        <Typography variant="h4" color="primary">
          Meus Modelos
        </Typography>
      </Stack>

      <Paper variant="outlined" sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom>
          Seus Modelos
        </Typography>
        <Typography variant="body1" color="text.secondary" gutterBottom>
          Visualize, treine e faça deploy dos modelos que você criou.
        </Typography>

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle1" gutterBottom>
          Ações disponíveis:
        </Typography>
        <Stack direction="row" spacing={2}>
          <Button variant="contained" color="primary">
            Treinar Novo
          </Button>
          <Button variant="contained" color="secondary">
            Remover Modelo
          </Button>
        </Stack>
      </Paper>
    </Box>
  );
}
